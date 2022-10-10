# -*- coding: utf-8 -*-
import pyqtgraph as pg

APP_NAME = 'opennft'
LOG_LEVEL = 'DEBUG'

ENVVAR_PREFIX = 'ONFT'

MAX_ROI_NAME_LENGTH = 6
PLOT_GRID_ALPHA = 0.7
ROI_PLOT_WIDTH = 2.0
MUSTER_Y_LIMITS = [-32767, 32768]
MUSTER_PLOT_ALPHA = 50

PROJ_ROI_COLORS = [
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
    (124, 196, 255, MUSTER_PLOT_ALPHA),
    (255, 156, 117, MUSTER_PLOT_ALPHA),
    (127, 255, 157, MUSTER_PLOT_ALPHA),
    (200, 200, 100, MUSTER_PLOT_ALPHA),
    (125, 125, 125, MUSTER_PLOT_ALPHA),
    (200, 100, 200, MUSTER_PLOT_ALPHA),
    (100, 200, 200, MUSTER_PLOT_ALPHA),
    (255, 22, 104, MUSTER_PLOT_ALPHA),
    (250, 104, 22, MUSTER_PLOT_ALPHA),
    (245, 245, 245, MUSTER_PLOT_ALPHA)
]