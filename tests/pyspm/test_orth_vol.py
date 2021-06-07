import numpy as np
from opennft import prepare_orth_view as pov
from opennft import update_orth_view as uov
from scipy.io import savemat


def test_orth_vol(data_path, nii_image_1, orth_matlab):

    mat = nii_image_1.affine
    dim = np. array(nii_image_1.shape)
    vol = np.array(nii_image_1.get_fdata(), order='F')

    str_param1 = pov.prepare_orth_view(mat, dim)

    imgt, imgc, imgs = uov.update_orth_view(str_param1, vol, mat)

    resl_dic = {"imgt_p": imgt, "imgc_p": imgc, "imgs_p": imgs}
    savemat(data_path / "orth_python.mat", resl_dic)

    np.testing.assert_almost_equal(imgt, orth_matlab['imgt'], decimal=0, err_msg="Not equal")
    print("\nimgt delta = %.4f" % ((imgt - orth_matlab['imgt']) ** 2).mean(axis=None))
    np.testing.assert_almost_equal(imgc, orth_matlab['imgc'], decimal=0, err_msg="Not equal")
    print("imgc delta = %.4f" % ((imgc - orth_matlab['imgc']) ** 2).mean(axis=None))
    np.testing.assert_almost_equal(imgs, orth_matlab['imgs'], decimal=0, err_msg="Not equal")
    print("imgs delta = %.4f" % ((imgs - orth_matlab['imgs']) ** 2).mean(axis=None))

    assert True, "Done"
