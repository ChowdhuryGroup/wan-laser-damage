import matplotlib.pyplot as plt
import numpy as np

photon_energy = 1.17  #
two_photon_absorb_coeff = 2.0e-11  # m W^-1, assuming 1060nm illumination
initial_carrier_density = 1.0e12 * (100**3)  # m^-3
initial_electron_temperature = initial_lattice_temperature = 300  # K

dt = 1.0e-17  # s
dx = dy = dz = 1.0e-9  # m


def divergence(vector_field: np.ndarray):
    # dimension = 3  # len(vector_field.shape) - 1
    result = np.zeros(vector_field.shape[:-1], dtype=np.float64)

    for i in range(vector_field.shape[0]):
        for j in range(vector_field.shape[1]):
            for k in range(vector_field.shape[2]):
                result[i, j, k] = (
                    vector_field[i + 1, j, k, 0] - vector_field[i - 1, j, k, 0]
                ) / dx
                result[i, j, k] += (
                    vector_field[i, j + 1, k, 1] - vector_field[i, j - 1, k, 1]
                ) / dy
                result[i, j, k] += (
                    vector_field[i, j, k + 1, 2] - vector_field[i, j, k - 1, 2]
                ) / dz
    return result


def gradient(scalar_field: np.ndarray):
    # dimension = 3  # len(scalar_field.shape)
    result = np.zeros((*(scalar_field.shape), 3))

    for i in range(scalar_field.shape[0]):
        for j in range(scalar_field.shape[1]):
            for k in range(scalar_field.shape[2]):
                result[i, j, k, 0] = (
                    scalar_field[i + 1, j, k] - scalar_field[i - 1, j, k]
                ) / dx
                result[i, j, k, 1] = (
                    scalar_field[i, j + 1, k] - scalar_field[i, j - 1, k]
                ) / dy
                result[i, j, k, 2] = (
                    scalar_field[i, j, k + 1] - scalar_field[i, j, k - 1]
                ) / dz
    return result


def ambipolar_diffusivity(lattice_temp):
    return 1.8e-3 * 300.0 / lattice_temp  # m^2 s^-1


def band_gap(lattice_temp, n):
    # Indirect band gap in eV
    # n is charge carriers per cubic meter
    result = (
        1.16
        - 7.02e-4 * lattice_temp**2 / (lattice_temp + 1108)
        - 1.5e-10 * n ** (1 / 3)
    )
    return result


def one_photon_absorb_coeff(T_l, n):
    # Also referred to as interband absorption
    E_g = band_gap(T_l, n)
    result = 6.0e5 * (
        (photon_energy - E_g - 0.0575) ** 2
        / (
            1
            - np.exp(-670 / T_l)
            + (photon_energy - E_g + 0.0575) ** 2 / (np.exp(670 / T_l) - 1)
        )
    )
    return result


def reflectivity(t: float):
    return 1.0


def update_n(n, I):
    n_dot = divergence(ambipolar_diffusivity(lattice_temp=1.0) * gradient(n))
    n_dot += one_photon_absorb_coeff + two_photon_absorb_coeff / 2.0 * I


def intensity_boundary_condition(r: float, t: float):
    F_0 = 1.0
    t_p = 1.0  # FWHM duration
    w_0 = 1.0
    return (
        (1 - reflectivity(t))
        * 1.763
        * F_0
        / (t_p * np.cosh(1.763 * t / t_p) ** 2)
        * np.exp(-((r / w_0) ** 2))
    )


if __name__ == "__main__":

    temps = np.linspace(2, 3.1, 100)
    temps = 10.0**temps
    n_vals = np.array([1.0e0, 1.0e3, 1.0e6, 1.0e9]) * initial_carrier_density

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Indirect Band Gap")
    axes[0].set_xlabel("Lattice Temperature / K")
    axes[0].set_ylabel("Band Gap / eV")
    axes[1].set_title("One Photon Absorption Coefficient (1064nm)")
    axes[1].set_xlabel("Lattice Temperature / K")
    axes[1].set_ylabel("Absorption Coefficient / m^-1")

    for n in n_vals:
        axes[0].semilogx(
            temps, band_gap(temps, n), label="n={:.3E} cm^-3".format(n / (100**3))
        )
        axes[1].loglog(
            temps,
            one_photon_absorb_coeff(temps, n),
            label="n={:.3E} cm^-3".format(n / (100**3)),
        )

    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    plt.show()
