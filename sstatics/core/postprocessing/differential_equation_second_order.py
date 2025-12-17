
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sstatics.core.preprocessing import BarSecond
from sstatics.core.postprocessing import DifferentialEquation


@dataclass(eq=False)
class DifferentialEquationSecond(DifferentialEquation):
    r"""Calculates discrete result vector for the provided bar using
    second-order analysis.

   Parameters
   ----------
   bar : :any:`BarSecond`
       The bar that is to be discretised.
   deform : :any:`numpy.ndarray`
       The deformation vector of the bar with shape (6, 1).
   forces : :any:`numpy.ndarray`
       The force vector of the bar with shape (6, 1).
   n_disc : :any:`int`, default=10
       Number of discrete points along the bar for discretisation.
   f_axial : :any:`float`, default=-1
        The axial force (:math:`L`) applied to the beam element, which is
        obtained from the internal force results of the first-order theory.

   Raises
   ------
   ValueError
       :py:attr:`deform` and :py:attr:`forces` need to have a shape
       of (6, 1)

   ValueError
       :py:attr:`n_disc` has to be greater than zero.

   ValueError
       :py:attr:`f_axial` can not be equal to zero.
   """
    f_axial: float = -1
    bar: BarSecond

    def __post_init__(self):
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        if self.f_axial == 0:
            raise ValueError('"f_axial" can not be equal to zero.')

    @cached_property
    def _c1(self):
        r"""Compute the unknown coefficient c_1 for the differential equation.

        Returns
        -------
        :any:`float`
            The coefficient c_1.

        Notes
        -----
        This coefficient depends on the transverse displacement at the start
        of the bar and c_3:

        - :math:`w_i`: Transverse displacement at the start of the bar.
        - :math:`c_3`: Unknown coefficient c_3.

        The formula is:

        .. math::
            c_1 = w_i - c_3
        """
        wi = self.deform[1][0]
        return wi - self._c3

    @cached_property
    def _c2(self):
        r"""
        Compute the unknown coefficient c_2 for the differential equation.

        Returns
        -------
        :any:`float`
            The coefficient c_2.

        Notes
        -----
        The coefficient depends on the slope at the start of the bar,
        bending moment, and distributed loads. The variables used are:

        - :math:`\ell` : Length of the bar
        - :math:`EI` : Flexural stiffness
        - :math:`B_s` : Modified flexural stiffness
        - :math:`GA_s` : Shear stiffness
        - :math:`\mu` : Characteristic number
        - :math:`\varphi_i` : Slope at start
        - :math:`p_{i,z}, p_{j,z}` : Distributed loads

        The formula is:

        .. math::
            c_2 = -\frac{\ell}{\mu} \cdot \varphi_i
                  - \left(\frac{B_s \cdot \mu}{GA_s \cdot \ell} +
                  \frac{\ell}{\mu}\right) \cdot c_{2,\text{first}}
                  + \text{factor} \cdot
                  \frac{EI \cdot \ell^2 \cdot (p_{j,z} - p_{i,z})}
                       {B_s \cdot GA_s \cdot \mu^3}

        with

        .. math::
            c_{2,\text{first}} = \frac{M_i}{EI} - \frac{q_i}{GA_s}

        and :math:`\text{factor} = \text{sign(f_axial)}`.
        """
        Bs, GAs, EI = self.bar.B_s, self.bar.GA_s, self.bar.EI
        l, mu = self.bar.length, self.bar.characteristic_number
        mi, phi_i = self.forces[2][0], self.deform[2][0]
        c2_first = mi / EI - self.bar.line_load[2][0] / GAs
        factor = np.sign(self.f_axial)
        return (- l/mu * phi_i - ((Bs * mu) / (GAs * l) + l / mu) * c2_first +
                factor * (EI * l ** 2 * (self.p_jz - self.p_iz)) /
                (Bs * GAs * mu ** 3))

    @cached_property
    def _c3(self):
        r"""Compute the unknown coefficient c_3 for the differential equation.

        Returns
        -------
        :any:`float`
            The coefficient c_3.

        Notes
        -----
        Variables used:

        - :math:`\ell`: Length of the bar.
        - :math:`EI`: Flexural stiffness.
        - :math:`B_s`: Modified flexural stiffness.
        - :math:`GA_s`: Shear stiffness.
        - :math:`\mu`: Characteristic number of the beam element.
        - :math:`M_i`: Bending moment at the start of the bar.
        - :math:`p_{i,z}`: Distributed load at the start of the bar.
        - :math:`\text{factor} = \text{sign(f_axial)}`

        The formula is:

        .. math::
            c_3 = \frac{\text{factor} \cdot M_i +
                     (\ell^2 / \mu^2 + EI / GA_s) \cdot p_{i,z}}{B_s}
                     \cdot \frac{\ell^2}{\mu^2}
            """
        Bs, GAs, EI = self.bar.B_s, self.bar.GA_s, self.bar.EI
        l, mu = self.bar.length, self.bar.characteristic_number
        mi = self.forces[2][0]
        factor = np.sign(self.f_axial)
        return ((factor * mi + (l ** 2 / (mu ** 2) + EI / GAs) * self.p_iz) *
                (l ** 2) / (Bs * (mu ** 2)))

    @cached_property
    def _c4(self):
        r"""Compute the unknown coefficient c_4 for the differential equation.

        Returns
        -------
        :any:`float`
            The coefficient c_4.

        Notes
        -----
        Variables used:

        - :math:`\ell`: Length of the bar.
        - :math:`EI`: Flexural stiffness.
        - :math:`B_s`: Modified flexural stiffness.
        - :math:`GA_s`: Shear stiffness.
        - :math:`\mu`: Characteristic number of the beam element.
        - :math:`V_i`: Shear force at the start of the bar.
        - :math:`p_{i,z}, p_{j,z}`: Distributed loads at start and end.
        - :math:`\text{factor} = \text{sign(f_axial)}`

        Formula:

        .. math::
            c_4 = (\text{factor} \cdot V_i +
                  (\ell / \mu^2 + EI / (GA_s \cdot \ell))
                  \cdot (p_{j,z} - p_{i,z}))
                  \cdot \frac{\ell^3}{B_s \cdot \mu^3}
        """
        Bs, GAs, EI = self.bar.B_s, self.bar.GA_s, self.bar.EI
        l, mu = self.bar.length, self.bar.characteristic_number
        vi = self.forces[1][0]
        factor = np.sign(self.f_axial)
        return (factor * vi + (l / (mu ** 2) + EI / (GAs * l)) *
                (self.p_jz - self.p_iz)) * l ** 3 / (Bs * mu ** 3)

    @cached_property
    def p_jz(self):
        r"""Compute the equivalent distributed load at the end of the bar along
        the curved axis.

        Returns
        -------
        :any:`float`
            Equivalent distributed load at the bar end.

        Notes
        -----
        Variables used:

        - :math:`\ell`: Length of the bar.
        - :math:`EI`: Flexural stiffness.
        - :math:`B_s`: Modified flexural stiffness.
        - :math:`GA_s`: Shear stiffness.
        - :math:`\mu`: Characteristic number of the beam element.
        - :math:`V_i, V_j`: Shear force at start and end of the bar.
        - :math:`M_i, M_j`: Bending moments at start and end of the bar.
        - :math:`p_{i,z}, p_{j,z}`: Distributed loads at start and end.
        - :math:`\text{factor} = \text{sign(f_axial)}`

        Formulas:

        For :math:`f_{\text{axial}} < 0`:

        .. math::
            p_{j,z} =
            \frac{- GA_s \cdot \mu \cdot ( -M_i \cdot \mu \cdot
            \cos(\mu) - M_j \cdot \mu + M_i \cdot \mu + M_j \cdot \mu
            \cdot \cos(\mu) - \ell \cdot V_i \cdot \sin(\mu) - \ell \cdot
            V_j \cdot \sin(\mu) + M_j \cdot \mu^2 \cdot \sin(\mu) + \ell
            \cdot \mu \cdot V_i + \ell \cdot \mu \cdot V_j \cdot \cos(\mu))}
            {(EI \cdot \mu^2 + GA_s \cdot \ell^2) \cdot (2 \cdot \cos(\mu)
            + \mu \cdot \sin(\mu) - 2)}

        For :math:`f_{\text{axial}} > 0`:

        .. math::
            p_{j,z} =
            \frac{- GA_s \cdot \mu \cdot ( M_j \cdot \mu - M_i \cdot \mu
            + M_i \cdot \mu \cdot \cosh(\mu) - M_j \cdot \mu \cdot \cosh(\mu)
            + \ell \cdot V_i \cdot \sinh(\mu) + \ell \cdot V_j \cdot \sinh(\mu)
            + M_j \cdot \mu^2 \cdot \sinh(\mu) - \ell \cdot \mu \cdot V_i
            - \ell \cdot \mu \cdot V_j \cdot \cosh(\mu))}
            {(EI \cdot \mu^2 - GA_s \cdot \ell^2) \cdot (\mu \cdot \sinh(\mu)
            - 2 \cdot \cosh(\mu) + 2)}
            """
        EI, GAs = self.bar.EI, self.bar.GA_s
        l, mu = self.bar.length, self.bar.characteristic_number
        mi, mj = self.forces[2][0], self.forces[5][0]
        vi, vj = self.forces[1][0], self.forces[4][0]
        mi_mu, mj_mu = mi * mu, mj * mu
        factor = np.sign(self.f_axial)

        if self.f_axial < 0:
            cos_mu, sin_mu = np.cos(mu), np.sin(mu)
            den = (2*cos_mu + mu * sin_mu - 2)
        else:
            cos_mu, sin_mu = np.cosh(mu), np.sinh(mu)
            den = (mu * sin_mu - 2 * cos_mu + 2)
        return (-(GAs * mu * (
                factor * mj_mu - factor * mi_mu + factor * mi_mu * cos_mu
                - factor * mj_mu * cos_mu + factor * l * vi * sin_mu
                + factor * l * vj * sin_mu + mj * mu ** 2 * sin_mu
                - factor * l * mu * vi - factor * l * mu * vj * cos_mu)) /
                ((EI * mu ** 2 - factor * GAs * l ** 2) * den))

    @cached_property
    def p_iz(self):
        r"""Compute the equivalent distributed load at the start of the bar
        along the curved axis.

        Returns
        -------
        :any:`float`
            Equivalent distributed load at the bar start.

        Notes
        -----
        Variables used:

        - :math:`\ell`: Length of the bar.
        - :math:`EI`: Flexural stiffness.
        - :math:`B_s`: Modified flexural stiffness.
        - :math:`GA_s`: Shear stiffness.
        - :math:`\mu`: Characteristic number of the beam element.
        - :math:`V_i, V_j`: Shear force at start and end of the bar.
        - :math:`M_i, M_j`: Bending moments at start and end of the bar.
        - :math:`p_{i,z}, p_{j,z}`: Distributed loads at start and end.
        - :math:`\text{factor} = \text{sign(f_axial)}`

        Formulas:

        For :math:`f_{\text{axial}} < 0`:

        .. math::
            p_{i,z} =
            \frac{- (EI \cdot p_{j,z} \cdot \mu^2 + GA_s \cdot \ell^2 \cdot
            p_{j,z} + GA_s \cdot \ell \cdot \mu^2 \cdot V_j - EI \cdot p_{j,z}
            \cdot \mu^2 \cdot \cos(\mu) - GA_s \cdot \ell^2 \cdot p_{j,z} \cdot
            \cos(\mu) - GA_s \cdot M_i \cdot \mu^3 \cdot \sin(\mu) + GA_s \cdot
            \ell \cdot \mu^2 \cdot V_i \cdot \cos(\mu))}
            {(EI \cdot \mu^2 + GA_s \cdot \ell^2) \cdot (\cos(\mu) +
            \mu \cdot \sin(\mu) - 1) \cdot \cosh(\mu) + 2}

        For :math:`f_{\text{axial}} > 0`:

        .. math::
            p_{i,z} =
            \frac{EI \cdot p_{j,z} \cdot \mu^2 - GA_s \cdot \ell^2 \cdot
            p_{j,z} + GA_s \cdot \ell \cdot \mu^2 \cdot V_j - EI \cdot p_{j,z}
            \cdot \mu^2 \cdot \cosh(\mu) + GA_s \cdot \ell^2 \cdot p_{j,z}
            \cdot \cosh(\mu) + GA_s \cdot M_i \cdot \mu^3 \cdot \sinh(\mu)
            + GA_s \cdot \ell \cdot \mu^2 \cdot V_i \cdot \cosh(\mu)}
            {(EI \cdot \mu^2 - GA_s \cdot \ell^2) \cdot (\mu \cdot \sinh(\mu)
            - \cosh(\mu) + 1)}
        """
        EI, GAs = self.bar.EI, self.bar.GA_s
        l, mu = self.bar.length, self.bar.characteristic_number
        mi = self.forces[2][0]
        vi, vj = self.forces[1][0], self.forces[4][0]
        factor = np.sign(self.f_axial)
        p_jz = self.p_jz

        if self.f_axial < 0:
            cos_mu, sin_mu = np.cos(mu), np.sin(mu)
            den = (cos_mu + mu * sin_mu-1)
        else:
            cos_mu, sin_mu = np.cosh(mu), np.sinh(mu)
            den = (mu * sin_mu - cos_mu + 1)
        return (factor * (EI * p_jz * mu ** 2 - factor * GAs * l ** 2 * p_jz
                          + GAs * l * mu ** 2 * vj
                          - EI * p_jz * mu ** 2 * cos_mu
                          + factor * GAs * l ** 2 * p_jz * cos_mu
                          + factor * GAs * mi * mu ** 3 * sin_mu
                          + GAs * l * mu ** 2 * vi * cos_mu) /
                ((EI * mu ** 2 - factor * GAs * l ** 2)*den))

    @cached_property
    def z_coef(self):
        r"""Compute the polynomial coefficients for the differential equation
        in the local z-direction.

        Returns
        -------
        :any:`numpy.ndarray`
            A (6, 5) matrix of polynomial coefficients. Rows correspond
            to the basis functions: x^3, x^2, x, sinh/cosh or sin/cos,
            constant term. Columns correspond to different physical
            quantities.

        Notes
        -----
        Variables used:

        - :math:`\ell`: Length of the bar.
        - :math:`EI`: Flexural stiffness.
        - :math:`B_s`: Modified flexural stiffness.
        - :math:`GA_s`: Shear stiffness.
        - :math:`\mu`: Characteristic number of the beam element.
        - :math:`V, M`: Shear force and bending moment at the start.
        - :math:`w, \varphi`: Transverse displacement and slope at start.
        - :math:`p_{i,z}, p_{j,z}`: Distributed load at start and end.
        - :math:`dp_z = p_{j,z} - p_{i,z}`.
        - :math:`\text{factor} = \text{sign(f_axial)}`.
        - :math:`cor`: Correction term to satisfy boundary conditions.

        Polynomial coefficients for L < 0:

        .. math::
            a_z =
            \begin{bmatrix}
            0 & -\dfrac{dp_z \cdot \ell}{\mu^2} - \dfrac{EI}{GA_s} \cdot
            \dfrac{dp_z}{\ell} & -\dfrac{p_{i,z} \cdot \ell^2}{\mu^2} -
            \dfrac{EI}{GA_s} \cdot p_{i,z} & c_2 & c_1 \\
            -B_s \cdot c_3 \cdot \dfrac{\mu^4}{\ell^4} & B_s \cdot c_4 \cdot
            \dfrac{\mu^3}{\ell^3} & B_s \cdot c_3 \cdot \dfrac{\mu^2}{\ell^2}
            & c_4 \cdot \dfrac{\mu}{\ell} & c_3 \\
            -B_s \cdot c_4 \cdot \dfrac{\mu^4}{\ell^4} & -B_s \cdot c_3 \cdot
            \dfrac{\mu^3}{\ell^3} & B_s \cdot c_4 \cdot \dfrac{\mu^2}{\ell^2}
            & -c_3 \cdot \dfrac{\mu}{\ell} & c_4 \\
            0 & 0 & -\dfrac{dp_z \cdot \ell}{\mu^2} - \dfrac{EI}{GA_s} \cdot
            \dfrac{dp_z}{\ell} & -\dfrac{p_{i,z} \cdot \ell^2}{2 \cdot B_s
            \cdot \mu^2} & c_2 + cor \\
            0 & 0 & 0 & -\dfrac{dp_z \cdot \ell}{2 \cdot B_s \cdot \mu^2} &
            \dfrac{p_{i,z} \cdot \ell^2}{2 \cdot B_s \cdot \mu^2} \\
            0 & 0 & 0 & 0 & \dfrac{dp_z \cdot \ell}{6 \cdot B_s \cdot \mu^2}
            \end{bmatrix}

        Polynomial coefficients for L > 0:

        .. math::
            a_z =
            \begin{bmatrix}
            0 & \dfrac{dp_z \cdot \ell}{\mu^2} - \dfrac{EI}{GA_s} \cdot
            \dfrac{dp_z}{\ell} & \dfrac{p_{i,z} \cdot \ell^2}{\mu^2} -
            \dfrac{EI}{GA_s} \cdot p_{i,z} & c_2 & c_1 \\
            B_s \cdot c_3 \cdot \dfrac{\mu^4}{\ell^4} & -B_s \cdot c_4 \cdot
            \dfrac{\mu^3}{\ell^3} & -B_s \cdot c_3 \cdot \dfrac{\mu^2}{\ell^2}
            & c_4 \cdot \dfrac{\mu}{\ell} & c_3 \\
            B_s \cdot c_4 \cdot \dfrac{\mu^4}{\ell^4} & -B_s \cdot c_3 \cdot
            \dfrac{\mu^3}{\ell^3} & -B_s \cdot c_4 \cdot \dfrac{\mu^2}{\ell^2}
            & -c_3 \cdot \dfrac{\mu}{\ell} & c_4 \\
            0 & 0 & \dfrac{dp_z \cdot \ell}{\mu^2} - \dfrac{EI}{GA_s} \cdot
            \dfrac{dp_z}{\ell} & -\dfrac{p_{i,z} \cdot \ell^2}{2 \cdot B_s
            \cdot \mu^2} & c_2 + cor \\
            0 & 0 & 0 & -\dfrac{dp_z \cdot \ell}{2 \cdot B_s \cdot \mu^2} &
            -\dfrac{p_{i,z} \cdot \ell^2}{2 \cdot B_s \cdot \mu^2} \\
            0 & 0 & 0 & 0 & -\dfrac{dp_z \cdot \ell}{6 \cdot B_s \cdot \mu^2}
            \end{bmatrix}
        """
        if np.isclose(self.f_axial, 0):
            return super().z_coef
        l, mu = self.bar.length, self.bar.characteristic_number
        Bs, GAs = self.bar.B_s, self.bar.GA_s
        EI = self.bar.EI
        dp_z = (self.p_jz - self.p_iz)
        p_iz = self.p_iz
        wj = self.deform[4][0]
        factor = np.sign(self.f_axial)
        if self.f_axial < 0:
            cos_mu, sin_mu = np.cos(mu), np.sin(mu)
            term = (-EI / GAs * dp_z / l)
        else:
            cos_mu, sin_mu = np.cosh(mu), np.sinh(mu)
            term = 0
        cor = (wj - (self._c1 + self._c2 * l + self._c3 * cos_mu
                     + self._c4 * sin_mu - factor * (self.p_iz * l ** 2) /
                     (2 * Bs * mu ** 2) * l ** 2 - factor * dp_z * l /
                     (6 * Bs * mu ** 2) * l ** 3)) / l
        return np.array([
            [0, factor * dp_z / mu ** 2 * l - EI / GAs * dp_z / l,
             factor * ((p_iz * l ** 2) / mu ** 2) - (EI * p_iz / GAs),
             self._c2, self._c1],
            [factor * Bs * self._c3 * mu ** 4 / l ** 4,
             - factor * Bs * self._c4 * mu ** 3 / l ** 3,
             - factor * Bs * self._c3 * mu ** 2 / l ** 2, self._c4 * mu / l,
             self._c3],
            [factor * Bs * self._c4 * mu ** 4 / l ** 4,
             -Bs * self._c3 * mu ** 3 / l ** 3,
             -factor * Bs * self._c4 * mu ** 2 / l ** 2, -self._c3 * mu / l,
             self._c4],
            [0, 0, term + factor * (dp_z * l / mu ** 2),
             -(self.p_iz * l ** 2) / (2 * Bs * mu ** 2), self._c2 + cor],
            [0, 0, 0, -(dp_z * l) / (2 * Bs * mu ** 2),
             - factor * self.p_iz * l ** 2 / (2 * Bs * mu ** 2)],
            [0, 0, 0, 0, - factor * (dp_z * l) / (6 * Bs * mu ** 2)]
        ])

    def _eval_poly(self, coef: np.ndarray):
        r"""
        Evaluate a (possibly trigonometric–polynomial) function along the
        bar using the given coefficient matrix.

        Parameters
        ----------
        coef : np.ndarray
            Coefficient matrix of shape (n, m), where
            - n: number of basis functions (≤ 6)
            - m: number of physical quantities (functions)

        Returns
        -------
        :any:`np.ndarray`
            Array of shape (k, m), where:
            - k: number of discretised positions along the bar
              (``len(self.x)``)
            - m: number of columns in ``coef``
            Each entry is a float value of the evaluated function.

        Notes
        -----
        The evaluation is performed using a trigonometric–polynomial
        basis, which depends on the number of coefficients provided:

        Basis functions:
            1 → :math:`1`
            2 → :math:`1, \cos(\mu / \ell \cdot x)`
            3 → :math:`1, \cos(\mu / \ell \cdot x),
            \sin(\mu / \ell \cdot x)`
            4 → :math:`1, \cos(\mu / \ell \cdot x),
            \sin(\mu / \ell \cdot x), x`
            5 → :math:`1, \cos(\mu / \ell \cdot x),
            \sin(\mu / \ell \cdot x), x, x^2`
            6 → :math:`1, \cos(\mu / \ell \cdot x),
            \sin(\mu / \ell \cdot x), x, x^2, x^3`

        Here, :math:`x` represents positions along the bar,
        :math:`\ell` is the length of the bar, and :math:`\mu` is the
        characteristic number. The trigonometric terms are selected based
        on the sign of the axial force:

        - For :math:`f_\text{axial} < 0`, use :math:`\cos` and :math:`\sin`.
        - For :math:`f_\text{axial} > 0`, use :math:`\cosh` and :math:`\sinh`.

        Each column of ``coef`` is multiplied by the corresponding basis
        vector, and the results are summed to produce the final function
        values along the discretised bar.
        """
        x = self.x
        l, mu = self.bar.length, self.bar.characteristic_number
        n_basis = len(coef)
        basis_columns = []

        if self.f_axial < 0:
            cos_mu = np.cos(mu / l * x)
            sin_mu = np.sin(mu / l * x)
        else:
            cos_mu = np.cosh(mu / l * x)
            sin_mu = np.sinh(mu / l * x)

        basis_columns.append(np.ones_like(x))
        if n_basis > 1:
            basis_columns.append(cos_mu)
        if n_basis > 2:
            basis_columns.append(sin_mu)
        if n_basis > 3:
            basis_columns.append(x)
        if n_basis > 4:
            basis_columns.append(x ** 2)
        if n_basis > 5:
            basis_columns.append(x ** 3)

        powers = np.column_stack(basis_columns)

        return powers @ coef
