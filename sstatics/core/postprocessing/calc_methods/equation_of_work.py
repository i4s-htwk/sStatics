from dataclasses import dataclass
import numpy as np
import pandas as pd
from sstatics.core.postprocessing.results import SystemResult

pd.options.display.float_format = "{:.6f}".format


@dataclass
class EquationOfWork:
    """Implements terms of the Equation of Work for real and virtual
    systems."""

    result_system_1: SystemResult   # reales System
    result_system_2: SystemResult   # virtuelles System

    def tab_moment(self) -> None:
        """Tabulate real and virtual bending moments at bar ends."""
        data = []
        for idx, (br_real, br_virt) in enumerate(
            zip(self.result_system_1.bars, self.result_system_2.bars),
            start=1,
        ):
            m_real = br_real.forces_disc[:, 2]
            m_virt = br_virt.forces_disc[:, 2]
            data.append([idx, m_real[0], m_real[-1], m_virt[0], m_virt[-1]])

        self.df = pd.DataFrame(
            data,
            columns=[
                "Stab", "M_real_start", "M_real_ende",
                "M_virt_start", "M_virt_ende",
            ],
        )
        print(self.df)

    def tab_shear(self) -> None:
        """Tabulate real and virtual shear forces at bar ends."""
        data = []
        for idx, (br_real, br_virt) in enumerate(
            zip(self.result_system_1.bars, self.result_system_2.bars),
            start=1,
        ):
            v_real = br_real.forces_disc[:, 1]
            v_virt = br_virt.forces_disc[:, 1]
            data.append([idx, v_real[0], v_real[-1], v_virt[0], v_virt[-1]])

        self.df = pd.DataFrame(
            data,
            columns=[
                "Stab", "V_real_start", "V_real_ende",
                "V_virt_start", "V_virt_ende",
            ],
        )
        print(self.df)

    def tab_v_coef_real(self) -> None:
        """Tabulate shear coefficients of the real system."""
        rows = []
        for idx, br in enumerate(self.result_system_1.bars, start=1):
            v_coef = br.z_coef[0:3, 1]
            rows.append([idx, *v_coef])

        self.df = pd.DataFrame(
            rows,
            columns=["Stab", "-V", "-p_i,z", "-dp_z/(2*l)"],
        )
        print(self.df)

    def tab_v_coef_virt(self) -> None:
        """Tabulate shear coefficients of the virtual system."""
        rows = []
        for idx, br in enumerate(self.result_system_2.bars, start=1):
            v_coef = br.z_coef[0:3, 1]
            rows.append([idx, *v_coef])

        self.df = pd.DataFrame(
            rows,
            columns=["Stab", "-V", "-p_i,z", "-dp_z/(2*l)"],
        )
        print(self.df)

    def tab_m_coef_real(self) -> None:
        """Tabulate bending moment coefficients of the real system."""
        rows = []
        for idx, br in enumerate(self.result_system_1.bars, start=1):
            m_coef = br.z_coef[0:4, 2]
            rows.append([idx, *m_coef])

        self.df = pd.DataFrame(
            rows,
            columns=["Stab", "-M", "-V", "-p_i,z/2", "-dp_z/(6*l)"],
        )
        print(self.df)

    def tab_m_coef_virt(self) -> None:
        """Tabulate bending moment coefficients of the virtual system."""
        rows = []
        for idx, br in enumerate(self.result_system_2.bars, start=1):
            m_coef = br.z_coef[0:4, 2]
            rows.append([idx, *m_coef])

        self.df = pd.DataFrame(
            rows,
            columns=["Stab", "-M", "-V", "-p_i,z/2", "-dp_z/(6*l)"],
        )
        print(self.df)

    def compute_moment_term(self) -> np.ndarray:
        """Compute bending moment contribution to the work equation."""
        momenten_anteil = np.zeros((len(self.result_system_1.bars), 1))
        for i, (br_real, br_virt) in enumerate(
            zip(self.result_system_1.bars, self.result_system_2.bars),
        ):
            length = br_real.bar.length
            ei = br_real.bar.EI

            z_coef_real = br_real.z_coef[0:4, 2]
            a1_real, a2_real, a3_real, a4_real = (
                z_coef_real[3], z_coef_real[2],
                z_coef_real[1], z_coef_real[0],
            )

            z_coef_virt = br_virt.z_coef[0:4, 2]
            a1_virt, a2_virt, a3_virt, a4_virt = (
                z_coef_virt[3], z_coef_virt[2],
                z_coef_virt[1], z_coef_virt[0],
            )

            momenten_anteil[i, 0] = (
                1 / ei * (
                    (a1_real * a1_virt / 7) * length**7
                    + (a1_real * a2_virt + a2_real * a1_virt) / 6 * length**6
                    + (a1_real * a3_virt + a2_real * a2_virt
                       + a3_real * a1_virt) / 5 * length**5
                    + (a1_real * a4_virt + a2_real * a3_virt
                       + a3_real * a2_virt + a4_real * a1_virt) / 4 * length**4
                    + (a2_real * a4_virt + a3_real * a3_virt
                       + a4_real * a2_virt) / 3 * length**3
                    + (a3_real * a4_virt + a4_real * a3_virt) / 2 * length**2
                    + (a4_real * a4_virt) * length
                )
            )
        return momenten_anteil

    def compute_shear_term(self) -> np.ndarray:
        """Compute shear force contribution to the work equation."""
        shear_term = np.zeros((len(self.result_system_1.bars), 1))
        for i, (br_real, br_virt) in enumerate(
            zip(self.result_system_1.bars, self.result_system_2.bars),
        ):
            length = br_real.bar.length
            kappa = br_real.bar.cross_section.shear_cor
            ga = (
                br_real.bar.material.shear_mod
                * br_real.bar.cross_section.area
            )

            v_coef_real = br_real.z_coef[0:3, 1]
            a1_real, a2_real, a3_real = (
                v_coef_real[2], v_coef_real[1], v_coef_real[0],
            )

            v_coef_virt = br_virt.z_coef[0:3, 1]
            a1_virt, a2_virt, a3_virt = (
                v_coef_virt[2], v_coef_virt[1], v_coef_virt[0],
            )

            shear_term[i, 0] = (
                kappa / ga * (
                    (a1_real * a1_virt) / 5 * length**5
                    + (a1_real * a2_virt + a2_real * a1_virt) / 4 * length**4
                    + (a1_real * a3_virt + a2_real * a2_virt
                       + a3_real * a1_virt) / 3 * length**3
                    + (a2_real * a3_virt + a3_real * a2_virt) / 2 * length**2
                    + (a3_real * a3_virt) * length
                )
            )
        return shear_term

    def compute_normal_term(self) -> np.ndarray:
        """Compute normal force contribution to the work equation."""
        normal_term = np.zeros((len(self.result_system_1.bars), 1))
        for i, (br_real, br_virt) in enumerate(
            zip(self.result_system_1.bars, self.result_system_2.bars),
        ):
            length = br_real.bar.length
            ea = br_real.bar.EA

            x_coef_real = br_real.x_coef[0:3, 1]
            a1_real, a2_real, a3_real = (
                x_coef_real[2], x_coef_real[1], x_coef_real[0],
            )

            x_coef_virt = br_virt.x_coef[0:3, 1]
            a1_virt, a2_virt, a3_virt = (
                x_coef_virt[2], x_coef_virt[1], x_coef_virt[0],
            )

            normal_term[i, 0] = (
                1 / ea * (
                    (a1_real * a1_virt) / 5 * length**5
                    + (a1_real * a2_virt + a2_real * a1_virt) / 4 * length**4
                    + (a1_real * a3_virt + a2_real * a2_virt
                       + a3_real * a1_virt) / 3 * length**3
                    + (a2_real * a3_virt + a3_real * a2_virt) / 2 * length**2
                    + (a3_real * a3_virt) * length
                )
            )
        return normal_term

    def compute_temperature_term(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute temperature load contribution to the work equation."""
        temp_const = np.zeros((len(self.result_system_1.bars), 1))
        temp_d_t = np.zeros((len(self.result_system_1.bars), 1))

        for i, (br_real, br_virt) in enumerate(
            zip(self.result_system_1.bars, self.result_system_2.bars),
        ):
            length = br_real.bar.length
            height = br_real.bar.cross_section.height
            alpha_t = br_real.bar.material.therm_exp_coeff

            t_s = br_real.bar.temp.temp_s
            d_t = br_real.bar.temp.temp_delta

            x_coef_virt = br_virt.x_coef[0:3, 1]
            n1_virt, n2_virt, n3_virt = (
                x_coef_virt[2], x_coef_virt[1], x_coef_virt[0],
            )

            z_coef_virt = br_virt.z_coef[0:4, 2]
            m1_virt, m2_virt, m3_virt, m4_virt = (
                z_coef_virt[3], z_coef_virt[2],
                z_coef_virt[1], z_coef_virt[0],
            )

            temp_const[i, 0] = t_s * alpha_t * (
                n1_virt / 3 * length**3
                + n2_virt / 2 * length**2
                + n3_virt * length
            )

            temp_d_t[i, 0] = d_t * alpha_t / height * (
                m1_virt / 4 * length**4
                + m2_virt / 3 * length**3
                + m3_virt / 2 * length**2
                + m4_virt * length
            )

        return temp_const, temp_d_t

    def delta_s1_s2(self) -> None:
        """Placeholder for combined work equation output."""
        return
