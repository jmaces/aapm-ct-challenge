from abc import ABC, abstractmethod

import numpy as np
import torch


# ----- Utilities -----


def l2_error(X, X_ref, relative=False, squared=False, use_magnitude=True):
    """ Compute average l2-error of an image over last three dimensions.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor of shape [..., 1, W, H] for real images or
        [..., 2, W, H] for complex images.
    X_ref : torch.Tensor
        The reference tensor of same shape.
    relative : bool, optional
        Use relative error. (Default False)
    squared : bool, optional
        Use squared error. (Default False)
    use_magnitude : bool, optional
        Use complex magnitudes. (Default True)

    Returns
    -------
    err_av :
        The average error.
    err :
        Tensor with individual errors.

    """
    assert X_ref.ndim >= 3  # do not forget the channel dimension

    if X_ref.shape[-3] == 2 and use_magnitude:  # compare complex magnitudes
        X_flat = torch.flatten(torch.sqrt(X.pow(2).sum(-3)), -2, -1)
        X_ref_flat = torch.flatten(torch.sqrt(X_ref.pow(2).sum(-3)), -2, -1)
    else:
        X_flat = torch.flatten(X, -3, -1)
        X_ref_flat = torch.flatten(X_ref, -3, -1)

    if squared:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1) ** 2
    else:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1)

    if relative:
        if squared:
            err = err / (X_ref_flat.norm(p=2, dim=-1) ** 2)
        else:
            err = err / X_ref_flat.norm(p=2, dim=-1)

    if X_ref.ndim > 3:
        err_av = err.sum() / np.prod(X_ref.shape[:-3])
    else:
        err_av = err
    return err_av.squeeze(), err


def fft1(x):
    """ 1-dimensional centered Fast Fourier Transform. """
    assert x.size(-1) == 2
    x = ifftshift(x, dim=(-2,))
    x = torch.fft(x, 1, normalized=True)
    x = fftshift(x, dim=(-2,))
    return x


def ifft1(x):
    """ 1-dimensional centered Inverse Fast Fourier Transform. """
    assert x.size(-1) == 2
    x = ifftshift(x, dim=(-2,))
    x = torch.ifft(x, 1, normalized=True)
    x = fftshift(x, dim=(-2,))
    return x


def roll(x, shift, dim):
    """ np.roll for torch tensors. """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """ np.fft.fftshift for torch tensors. """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [xdim // 2 for xdim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """ np.fft.ifftshift for torch tensors. """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(xdim + 1) // 2 for xdim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


# ----- Linear Operator Utilities -----


class LinearOperator(ABC):
    """ Abstract base class for linear (measurement) operators.

    Can be used for real operators

        A : R^(n1 x n2) ->  R^m

    or complex operators

        A : C^(n1 x n2) -> C^m.

    Can be applied to tensors of shape (n1, n2) or (1, n1, n2) or batches
    thereof of shape (*, n1, n2) or (*, 1, n1, n2) in the real case, or
    analogously shapes (2, n1, n2) or (*, 2, n1, n2) in the complex case.

    Attributes
    ----------
    m : int
        Dimension of the co-domain of the operator.
    n : tuple of int
        Dimensions of the domain of the operator.

    """

    def __init__(self, m, n):
        self.m = m
        self.n = n

    @abstractmethod
    def dot(self, x):
        """ Application of the operator to a vector.

        Computes Ax for a given vector x from the domain.

        Parameters
        ----------
        x : torch.Tensor
            Must be of shape to (*, n1, n2) or (*, 2, n1, n2).
        Returns
        -------
        torch.Tensor
            Will be of shape (*, m) or (*, 2, m).
        """
        pass

    @abstractmethod
    def adj(self, y):
        """ Application of the adjoint operator to a vector.

        Computes (A^*)y for a given vector y from the co-domain.

        Parameters
        ----------
        y : torch.Tensor
            Must be of shape (*, m) or (*, 2, m).

        Returns
        -------
        torch.Tensor
            Will be of shape (*, n1, n2) or (*, 2, n1, n2).
        """
        pass

    @abstractmethod
    def inv(self, y):
        """ Application of some inversion of the operator to a vector.

        Computes (A^dagger)y for a given vector y from the co-domain.
        A^dagger can for example be the pseudo-inverse.

        Parameters
        ----------
        y : torch.Tensor
            Must be of shape (*, m) or (*, 2, m).

        Returns
        -------
        torch.Tensor
            Will be of shape (*, n1, n2) or (*, 2, n1, n2).
        """
        pass

    def __call__(self, x):  # alias to make operator callable by using dot
        return self.dot(x)


# ----- Measurement Operators -----


class FanbeamRadon(torch.nn.Module, LinearOperator):
    """ Parametrized implementation of a discrete  fanbeam Radon transform.

    Allows for automatic differentiation with respect to the parameters
    defining the fanbeam geometry.

    Parameters
    ----------
    n : (int, int)
        Dimensions of the discrete image signals. Currently only tested on
        square images, so n=(n1, n2) should satisfy n1=n2.
    angles : torch.Tensor
        Array of fanbeam rotation angles in the range [0..360].
    d_source : torch.Tensor
        Distance of X-ray source to origin (center of rotation). The distance
        of the origin to the detector array is inferred from field-of-view
        angle, which is chosen so that the maximum inscribed circle in the
        image signal lies exactly within each fan.
    n_detect : int
        Number of detectors in the detector array.
    s_detect : torch.Tensor
        Spacing of detectors along the detector array.
    flat : bool
        Use a flat or curved detector array. (Default True)
    filter_type : str
        Filter to use for the filtered backprojection (inversion). One of
        ["hamming", "cosine", "hann", "ramp"]. (Default "hamming")
    learn_inv_scale : bool
        Make scalingf factor for the filtered backprojection (inversion)
        learnable. (Default False)
    """

    def __init__(
        self,
        n,
        angles,
        scale,
        d_source,
        n_detect,
        s_detect,
        flat=True,
        filter_type="hamming",
        learn_inv_scale=False,
    ):
        super(FanbeamRadon, self).__init__()
        self.n = torch.nn.Parameter(torch.tensor(n), requires_grad=False)
        self.m = torch.nn.Parameter(
            torch.tensor((len(angles), n_detect)), requires_grad=False
        )
        self.angles = torch.nn.Parameter(
            angles, requires_grad=angles.requires_grad
        )
        self.scale = torch.nn.Parameter(
            scale, requires_grad=scale.requires_grad
        )
        self.d_source = torch.nn.Parameter(
            d_source, requires_grad=d_source.requires_grad
        )
        self.n_detect = torch.nn.Parameter(
            torch.tensor(n_detect), requires_grad=False
        )
        self.s_detect = torch.nn.Parameter(
            s_detect, requires_grad=s_detect.requires_grad
        )
        self.fwd_offset = torch.nn.Parameter(
            torch.zeros(*self.m), requires_grad=False,
        )
        self.flat = flat
        self.filter_type = filter_type
        self.inv_scale = torch.nn.Parameter(
            torch.tensor(1.0), requires_grad=learn_inv_scale
        )

    def _d_detect(self):
        if self.flat:
            return (
                abs(self.s_detect)
                * self.n_detect
                / self.n[0]
                * torch.sqrt(
                    self.d_source * self.d_source
                    - (self.n[0] / 2.0) * (self.n[0] / 2.0)
                )
                - self.d_source
            )
        else:
            return (
                abs(self.s_detect)
                * (self.n_detect / 2.0)
                / torch.asin((self.n[0] / 2.0) / self.d_source)
                - self.d_source
            )

    def forward(self, x):
        return self.dot(x)

    def dot(self, x):
        # detector positions
        s_range = (
            torch.arange(self.n_detect, device=self.n_detect.device).unsqueeze(
                0
            )
            - self.n_detect / 2.0
            + 0.5
        ) * self.s_detect
        if self.flat:
            p_detect_x = s_range
            p_detect_y = -self._d_detect()
        else:
            gamma = s_range / (self.d_source + self._d_detect())
            p_detect_x = (self.d_source + self._d_detect()) * torch.sin(gamma)
            p_detect_y = self.d_source - (
                self.d_source + self._d_detect()
            ) * torch.cos(gamma)

        # source position
        p_source_x = 0.0
        p_source_y = self.d_source

        # rotate rays from source to detector over all angles
        pi = torch.acos(torch.zeros(1)).item() * 2.0
        cs = torch.cos(self.angles * pi / 180.0).unsqueeze(1)
        sn = torch.sin(self.angles * pi / 180.0).unsqueeze(1)
        r_p_source_x = p_source_x * cs - p_source_y * sn
        r_p_source_y = p_source_x * sn + p_source_y * cs
        r_dir_x = p_detect_x * cs - p_detect_y * sn - r_p_source_x
        r_dir_y = p_detect_x * sn + p_detect_y * cs - r_p_source_y

        # find intersections of rays with circle for clipping
        if self.flat:
            max_gamma = torch.atan(
                (self.s_detect.abs() * (self.n_detect / 2.0))
                / (self.d_source + self._d_detect())
            )
        else:
            max_gamma = (self.s_detect.abs() * (self.n_detect / 2.0)) / (
                self.d_source + self._d_detect()
            )
        radius = self.d_source * torch.sin(max_gamma)
        a = r_dir_x * r_dir_x + r_dir_y * r_dir_y
        b = r_p_source_x * r_dir_x + r_p_source_y * r_dir_y
        c = (
            r_p_source_x * r_p_source_x
            + r_p_source_y * r_p_source_y
            - radius * radius
        )
        ray_length_threshold = 1.0
        discriminant_sqrt = torch.sqrt(
            torch.max(
                b * b - a * c,
                torch.tensor(ray_length_threshold, device=x.device),
            )
        )
        lambda_1 = (-b - discriminant_sqrt) / a
        lambda_2 = (-b + discriminant_sqrt) / a

        # clip ray accordingly
        r_p_source_x = r_p_source_x + lambda_1 * r_dir_x
        r_p_source_y = r_p_source_y + lambda_1 * r_dir_y
        r_dir_x = r_dir_x * (lambda_2 - lambda_1)
        r_dir_y = r_dir_y * (lambda_2 - lambda_1)

        # use batch and channel dimensions for vectorized interpolation
        original_dim = x.ndim
        while x.ndim < 4:
            x = x.unsqueeze(0)
        assert x.shape[-3] == 1  # we can handle only single channel data
        x = x.transpose(-4, -3)  # switch batch and channel dim

        # integrate over ray
        num_steps = torch.ceil(
            torch.sqrt(r_dir_x * r_dir_x + r_dir_y * r_dir_y)
        ).max()
        diff_x = r_dir_x / num_steps
        diff_y = r_dir_y / num_steps
        steps = (
            torch.arange(
                int(num_steps.detach().cpu().numpy()), device=x.device
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        grid_x = r_p_source_x.unsqueeze(0) + steps * diff_x.unsqueeze(0)
        grid_y = r_p_source_y.unsqueeze(0) + steps * diff_y.unsqueeze(0)

        grid_x = grid_x / (
            self.n[0] / 2.0 - 0.5
        )  # rescale image positions to [-1, 1]
        grid_y = grid_y / (
            self.n[1] / 2.0 - 0.5
        )  # rescale image positions to [-1, 1]
        grid = torch.stack([grid_y, grid_x], dim=-1)
        inter = torch.nn.functional.grid_sample(
            x.expand((int(num_steps.detach().cpu().numpy()), -1, -1, -1)),
            grid,
            align_corners=True,
        )

        sino = inter.sum(dim=0, keepdim=True) * torch.sqrt(
            diff_x * diff_x + diff_y * diff_y
        ).unsqueeze(0).unsqueeze(0)

        # undo batch and channel manipulations
        sino = sino.transpose(-4, -3)  # unswitch batch and channel dim
        while sino.ndim > original_dim:
            sino = sino.squeeze(0)

        return sino * self.scale + self.fwd_offset

    def _adj(self, sino):
        """ Basic back projection without filtering or pre weighting. """
        # image coordinate grid
        p_x = torch.linspace(
            -self.n[0] / 2.0 + 0.5,
            self.n[0] / 2.0 - 0.5,
            self.n[0],
            device=self.n.device,
        ).unsqueeze(1)
        p_y = torch.linspace(
            -self.n[1] / 2.0 + 0.5,
            self.n[1] / 2.0 - 0.5,
            self.n[1],
            device=self.n.device,
        ).unsqueeze(0)

        # check if coordinate is within circle
        if self.flat:
            max_gamma = torch.atan(
                (self.s_detect.abs() * (self.n_detect / 2.0))
                / (self.d_source + self._d_detect())
            )
        else:
            max_gamma = (self.s_detect.abs() * (self.n_detect / 2.0)) / (
                self.d_source + self._d_detect()
            )
        radius = self.d_source * torch.sin(max_gamma)
        p_r = torch.sqrt(p_x * p_x + p_y * p_y)
        mask = p_r <= radius

        # use batch and channel dimensions for vectorized interpolation
        original_dim = sino.ndim
        while sino.ndim < 4:
            sino = sino.unsqueeze(0)
        assert sino.shape[-3] == 1  # we can handle only single channel data
        sino = sino.transpose(-4, -3)  # switch batch and channel dim

        # rotated coordinate grid
        pi = torch.acos(torch.zeros(1)).item() * 2.0
        cs = torch.cos(self.angles * pi / 180.0).unsqueeze(1).unsqueeze(1)
        sn = torch.sin(self.angles * pi / 180.0).unsqueeze(1).unsqueeze(1)
        p_x_r = cs * p_x + sn * p_y
        p_y_r = -sn * p_x + cs * p_y

        # find angles and detector positions defining rays through coordinate
        if self.flat:
            grid_d = (
                (self.d_source + self._d_detect())
                * p_x_r
                / (self.d_source - p_y_r)
            )
        else:
            grid_d = (self.d_source + self._d_detect()) * torch.atan(
                p_x_r / (self.d_source - p_y_r)
            )
        grid_a = (
            torch.arange(self.m[0], device=sino.device)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self.n[0], self.n[1])
            - self.m[0] / 2.0
            + 0.5
        )

        grid_d = grid_d / (
            (self.n_detect / 2.0 - 0.5) * self.s_detect
        )  # rescale valid detector positions to [-1,1]
        grid_a = grid_a / (self.m[0] / 2.0 - 0.5)  # rescale angles to [-1,1]
        grid = torch.stack([grid_d, grid_a], dim=-1)
        inter = torch.nn.functional.grid_sample(
            sino.expand(self.m[0], -1, -1, -1), grid, align_corners=True
        )

        # compute integral reweighting factors and integrate
        if self.flat:
            weight = (self.d_source + self._d_detect()).pow(2) / (
                self.d_source - p_y_r
            ).pow(2)
        else:
            weight = (self.d_source + self._d_detect()).pow(2) / (
                (self.d_source - p_y_r).pow(2) + p_x_r.pow(2)
            )
        x = mask * (inter * (weight).unsqueeze(1)).sum(dim=0, keepdim=True)

        # undo batch and channel manipulations
        x = x.transpose(-4, -3)  # unswitch batch and channel dim
        while x.ndim > original_dim:
            x = x.squeeze(0)

        return x / self.s_detect.abs()

    def _reweight_sinogram(self, sino):
        """ Reweight sinogram contributions to back projections. """
        return sino * self._get_pre_weight()

    def _filter_sinogram(self, sino):
        """ Pad and filter sinogram. """
        # pad sinogram to reduce periodicity artefacts
        target_size = max(
            64, int(2 ** np.ceil(np.log2(2 * self.m[-1].item())))
        )
        pad = target_size - self.m[-1]
        sino_pad = torch.nn.functional.pad(sino, (pad // 2, pad - pad // 2))

        # fft along detector direction
        sino_complex = torch.stack(
            [sino_pad, torch.zeros_like(sino_pad)], dim=-1
        )
        sino_fft = fft1(sino_complex)

        # apply frequency filter
        f = self._get_fourier_filter().unsqueeze(-1).to(sino.device)
        filtered_sino_fft = sino_fft * f

        # ifft along detector direction
        filtered_sino_pad = ifft1(filtered_sino_fft)[..., 0]

        # remove padding and rescale
        filtered_sino = filtered_sino_pad[..., pad // 2 : -(pad - pad // 2)]

        return filtered_sino

    def adj(self, sino):
        """ Unfiltered back projection (approx. adjoint). """
        sino = self._reweight_sinogram(sino * self.scale)
        return self._adj(sino)  # * self.scale

    def inv(self, sino):
        """ Filtered back projection (FBP). """
        sino = self._reweight_sinogram(sino / self.scale)
        sino = self._filter_sinogram(sino)
        return self.inv_scale * self._adj(sino)  # / self.scale

    def _get_fourier_filter(self):
        """ Ramp Fourier filter for the FBP. """
        size = max(64, int(2 ** np.ceil(np.log2(2 * self.m[-1].item()))))

        pi = torch.acos(torch.zeros(1)).item() * 2.0
        n = torch.cat(
            [
                torch.arange(1, size // 2 + 1, 2, device=self.n.device),
                torch.arange(size // 2 - 1, 0, -2, device=self.n.device),
            ]
        )
        f = torch.zeros(size, device=self.n.device)
        f[0] = 0.25
        if self.flat:
            f[1::2] = -1 / (pi * n).pow(2)
        else:
            f[1::2] = -self.s_detect.abs().pow(2) / (
                pi
                * (self.d_source + self._d_detect())
                * torch.sin(
                    n
                    * self.s_detect.abs()
                    / (self.d_source + self._d_detect())
                )
            ).pow(2)
        f = torch.stack(
            [f, torch.zeros(f.shape, device=self.n.device)], dim=-1
        )
        f = fftshift(f, dim=(-2,))

        filt = fft1(f)[..., 0]

        if self.filter_type == "hamming":
            # hamming filter
            fac = torch.tensor(
                np.hamming(size).astype(np.float32), device=f.device
            )
        elif self.filter_type == "hann":
            # hann filter
            fac = torch.tensor(
                np.hanning(size).astype(np.float32), device=f.device
            )
        elif self.filter_type == "cosine":
            # cosine filter
            fac = torch.sin(
                torch.linspace(0, pi, size + 1, device=f.device)[:-1]
            )
        else:
            # ramp / ram-lak filter
            fac = 1.0

        return fac * filt

    def _get_pre_weight(self):
        """ Pre filtering weighting for back projections. """
        s_range = (
            torch.arange(self.n_detect, device=self.n_detect.device).unsqueeze(
                0
            )
            - self.n_detect / 2.0
            + 0.5
        ) * self.s_detect
        if self.flat:
            weight = self.d_source / torch.sqrt(
                (self.d_source + self._d_detect()).pow(2) + s_range.pow(2)
            )
        else:
            weight = (
                self.d_source
                / (self.d_source + self._d_detect())
                * torch.cos(s_range / (self.d_source + self._d_detect()))
            )
        return weight
