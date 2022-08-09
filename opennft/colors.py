import pyqtgraph as pg
from opennft.config import config as con

ROI_PLOT_COLORS = [
    pg.mkColor(0, 0, 255, 255),
    pg.mkColor(0, 255, 255, 255),
    pg.mkColor(0, 255, 0, 255),
    pg.mkColor(255, 0, 255, 255),
    pg.mkColor(255, 0, 0, 255),
    pg.mkColor(255, 255, 0, 255),
    pg.mkColor(140, 200, 240, 255),
    pg.mkColor(208, 208, 147, 255),
    pg.mkColor(147, 0, 0, 255),
    pg.mkColor(100, 175, 0, 255),
    pg.mkColor(147, 255, 0, 255),
    pg.mkColor(120, 147, 147, 255)
]

MUSTER_PEN_COLORS = [
    (73, 137, 255, 255),
    (255, 103, 86, 255),
    (22, 255, 104, 255),
    (200, 200, 100, 255),
    (125, 125, 125, 255),
    (200, 100, 200, 255),
    (100, 200, 200, 255),
    (255, 22, 104, 255),
    (250, 104, 22, 255),
    (245, 245, 245, 255)
]
MUSTER_BRUSH_COLORS = [
    (124, 196, 255, con.muster_plot_alpha),
    (255, 156, 117, con.muster_plot_alpha),
    (127, 255, 157, con.muster_plot_alpha),
    (200, 200, 100, con.muster_plot_alpha),
    (125, 125, 125, con.muster_plot_alpha),
    (200, 100, 200, con.muster_plot_alpha),
    (100, 200, 200, con.muster_plot_alpha),
    (255, 22, 104, con.muster_plot_alpha),
    (250, 104, 22, con.muster_plot_alpha),
    (245, 245, 245, con.muster_plot_alpha)
]