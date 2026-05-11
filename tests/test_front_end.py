import numpy as np

from dasvo.front_end import KLTFrontEnd, ORBFrontEnd


def test_klt_frontend():
    frontend = KLTFrontEnd()
    
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    pts1, pts2 = frontend.process_frame(img1)
    assert pts1 is None
    assert pts2 is None
    
    pts1, pts2 = frontend.process_frame(img2)
    # Since images are black, no features should be found
    assert pts1 is None
    assert pts2 is None


def test_orb_frontend():
    frontend = ORBFrontEnd()
    
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    pts1, pts2 = frontend.process_frame(img1)
    assert pts1 is None
    assert pts2 is None
    
    pts1, pts2 = frontend.process_frame(img2)
    # Since images are black, no features should be found
    assert pts1 is None
    assert pts2 is None
