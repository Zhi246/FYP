"""Video-encoding unit tests only; artificial arrays are not experiment data."""
import unittest
import numpy as np
from extract_kth_spatiotemporal import encode_frames, independent_encode_u8


class EncodingTests(unittest.TestCase):
    def test_zero_motion(self):
        old,new=encode_frames(np.full((17,64,64),.4,dtype=np.float32))
        self.assertEqual(new.shape,(16,));np.testing.assert_array_equal(old,0);np.testing.assert_array_equal(new,0)

    def test_single_region_first_interval(self):
        frames=np.zeros((17,64,64),dtype=np.float32);frames[1:,:32,:32]=1
        old,new=encode_frames(frames)
        expected=np.zeros(16);expected[0]=.25
        np.testing.assert_array_equal(new,expected);np.testing.assert_array_equal(old,expected)

    def test_region_major_time_order(self):
        frames=np.zeros((17,64,64),dtype=np.float32)
        regions=[(0,32,0,32),(0,32,32,64),(32,64,0,32),(32,64,32,64)]
        for interval in range(16):
            frames[interval+1]=frames[interval]
            r=interval//4;y0,y1,x0,x1=regions[r]
            frames[interval+1,y0:y1,x0:x1]+=(r+1)*.01
        _,new=encode_frames(frames)
        np.testing.assert_allclose(new.reshape(4,4),np.diag([.01,.02,.03,.04]),atol=1e-8,rtol=1e-6)

    def test_float_subtraction_and_uint8_rejection(self):
        frames=np.zeros((17,64,64),dtype=np.uint8);frames[0]=255
        with self.assertRaises(ValueError):encode_frames(frames)
        old,new=encode_frames(frames.astype(np.float32)/255)
        self.assertEqual(old[0],1.)
        expected=np.zeros((4,4));expected[:,0]=.25
        np.testing.assert_array_equal(new.reshape(4,4),expected)

    def test_independent_float64_definition(self):
        gray=np.random.default_rng(42).integers(0,256,size=(17,64,64),dtype=np.uint8)
        old,new=encode_frames(gray.astype(np.float32)/255.)
        reference_old,reference_new=independent_encode_u8(gray)
        np.testing.assert_allclose(old,reference_old,atol=1e-7,rtol=0)
        np.testing.assert_allclose(new,reference_new,atol=1e-7,rtol=0)

    def test_average_conservation(self):
        frames=np.random.default_rng(42).random((17,64,64),dtype=np.float32)
        old,new=encode_frames(frames)
        np.testing.assert_allclose(new.reshape(4,4).mean(axis=0),old.reshape(4,4).mean(axis=1),atol=1e-7,rtol=0)

    def test_invalid_shape_nan_and_range(self):
        for frames in [np.zeros((16,64,64)),np.full((17,64,64),np.nan),np.full((17,64,64),2.)]:
            with self.assertRaises(ValueError):encode_frames(frames)


if __name__=="__main__":unittest.main()
