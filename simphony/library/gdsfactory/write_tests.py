from simphony.library.gdsfactory import _elements


def write_test_sparameters():
    """ writes a regression test for all the component properties dict"""
    with open("test_components.py", "w") as f:
        f.write("# this code has been automatically generated from write_tests.py\n")
        f.write("import numpy as np\n\n")
        f.write("import simphony.library.gdsfactory as cl\n\n")

        for c in _elements:
            f.write(
                f"""
def test_{c}(data_regression):
    c = cl.{c}()
"""
                + """
    wav = np.linspace(1520, 1570, 3) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)
    _, rows, cols = np.shape(s)
    sdict = {f'S{i+1}{j+1}': np.abs(s[:, i, j]).tolist() for i in range(rows) for j in range(cols)}
    data_regression.check(sdict)

"""
            )


if __name__ == "__main__":
    write_test_sparameters()
