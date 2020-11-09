import numpy as np

BACK = {
    "K": np.matrix([[233.8023948358759, 0.0, 319.18098083378686],
                    [0.0, 235.029598814203, 230.0816837094292],
                    [0.0, 0.0, 1.0]]),
    "D": np.matrix([[-0.03728316338273715], [0.064970747106063], [-0.1280365744356986], [0.0746922030634132]]),
    "R": np.matrix([[-0.014438624439362878, 0.999703156763941, 0.019624588671109708],
                    [-0.6749437018426232, -0.024225023632511857, 0.7374714554292799],
                    [0.7377279481402851, -0.002597419145123949, 0.675093125388275]]),
    "T": np.matrix([[0.02805094069587906],
                    [-0.19990103713561336],
                    [-0.27215031453153804]]),
}
FRONT = {
    "K": np.matrix([[215.55625191848213, 0.0, 329.098771437234],
                    [0.0, 216.61185821474132, 231.26915592635174],
                    [0.0, 0.0, 1.0]]),
    "D": np.matrix([[0.039869449286715763], [-0.013864212291013613], [-0.029857251916734356], [0.033996487636925955]]),
    "R": np.matrix([[0.9989189807633628, 0.030522741905540975, -0.03506040640454697],
                    [0.00638435077226743, 0.6569978971275666, 0.7538653747421831],
                    [0.04604465154500481, -0.7532742697029571, 0.6560928018714851]]),
    "T": np.matrix([[0.34575648899297173],
                    [0.6318473674545044],
                    [-1.1893240062570378]]),
}
LEFT = {
    "K": np.matrix([[238.34997484819695, 0.0, 326.53544153974417],
                    [0.0, 238.81386063611134, 240.7183666495277],
                    [0.0, 0.0, 1.0]]),
    "D": np.matrix([[-0.029966485422785252], [-0.022588531471370504], [0.049646873062577354], [-0.04000003516378078]]),
    "R": np.matrix([[-0.015779411256102317, -0.9998316467665483, 0.009364203356554161],
                    [0.6642976589352398, -0.0034834447499594735, 0.7474600229750309],
                    [-0.7473015659782672, 0.018015097467571108, 0.6642407889836812]]),
    "T": np.matrix([[-0.009003558976260678],
                    [0.09783723379003023],
                    [-0.5618371686522383]]),
}
RIGHT = {
    "K": np.matrix([[236.63876861800844, 0.0, 318.10479837173074],
                    [0.0, 236.02443896390108, 243.20083824811755],
                    [0.0, 0.0, 1.0]]),
    "D": np.matrix([[-0.035914829448729285], [0.011278361782286755], [-0.017546593017396903], [0.006029226403845235]]),
    "R": np.matrix([[-0.9993559673851283, -0.012636582170090222, -0.033585223578127404],
                    [-0.013830699475538713, -0.7279865727082687, 0.6854518668064782],
                    [-0.03311136064482372, 0.685474920582458, 0.7273429528418632]]),
    "T": np.matrix([[-0.17486764291504775],
                    [0.8576764216752175],
                    [-1.112007297629614]]),
}

ORIGINAL_RESOLUTION = (1280, 720)
TARGET_RESOLUTION = (1000, 1000)
