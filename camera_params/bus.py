import numpy as np

BACK = {
    "K": np.matrix([[488.0964314061363, 0.0, 630.609721823267],
                    [0.0, 490.0084929066934, 392.771171180756],
                    [0.0, 0.0, 1.0]]),
    "D": np.matrix([0., 0., 0., 0.]),
    "T": np.matrix([[0.9989189807633628, 0.030522741905540975, -0.03506040640454697, 0.34575648899297173],
                    [0.00638435077226743, 0.6569978971275666, 0.7538653747421831, 0.6318473674545044],
                    [0.04604465154500481, -0.7532742697029571, 0.6560928018714851, -1.1893240062570378],
                    [0.0, 0.0, 0.0, 1.]]).I
}
FRONT = {
    "K": np.matrix([[499.2101003649497, 0.0, 646.5405822453878],
                    [0.0, 500.61593672487885, 334.76781996287565],
                    [0.0, 0.0, 1.0]]),
    "D": np.matrix([0., 0., 0., 0.]),
    "T": np.matrix([[-0.9993559673851283, -0.012636582170090222, -0.033585223578127404, -0.17486764291504775],
                    [-0.013830699475538713, -0.7279865727082687, 0.6854518668064782, 0.8576764216752175],
                    [-0.03311136064482372, 0.685474920582458, 0.7273429528418632, -1.112007297629614],
                    [0.0, 0.0, 0.0, 1.]]).I
}
LEFT = {
    "K": np.matrix([[497.02280313949956, 0.0, 648.0150740277469],
                    [0.0, 498.62091488827565, 360.9442585112354],
                    [0.0, 0.0, 1.0]]),
    "D": np.matrix([0., 0., 0., 0.]),
    "T": np.matrix([[-0.015779411256102317, -0.9998316467665483, 0.009364203356554161, -0.009003558976260678],
                    [0.6642976589352398, -0.0034834447499594735, 0.7474600229750309, 0.09783723379003023],
                    [-0.7473015659782672, 0.018015097467571108, 0.6642407889836812, -0.5618371686522383],
                    [0.0, 0.0, 0.0, 1.]]).I
}
RIGHT = {
    "K": np.matrix([[496.8355185092727, 0.0, 645.5095173768549],
                    [0.0, 498.9636056851224, 354.4134422810794],
                    [0.0, 0.0, 1.0]]),
    "D": np.matrix([0., 0., 0., 0.]),
    "T": np.matrix([[-0.014438624439362878, 0.999703156763941, 0.019624588671109708, 0.02805094069587906],
                    [-0.6749437018426232, -0.024225023632511857, 0.7374714554292799, -0.19990103713561336],
                    [0.7377279481402851, -0.002597419145123949, 0.675093125388275, -0.27215031453153804],
                    [0.0, 0.0, 0.0, 1.]]).I

}

ORIGINAL_RESOLUTION = (1280, 720)
TARGET_RESOLUTION = (1000, 1000)
