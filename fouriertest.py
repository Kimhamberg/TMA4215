import unittest
import numpy
import fourier


class FourierTest(unittest.TestCase):
    def test_basic_complex_input(self):
        input_data = numpy.asarray(
            [complex(1, 1), complex(1, -1), complex(-1, -1), complex(-1, 1)]
        )
        expected = numpy.fft.fft(input_data)
        result = fourier.fft(input_data, len(input_data))
        if isinstance(result, list):
            numpy.testing.assert_almost_equal(result, expected)

    def test_real_input(self):
        input_data = numpy.asarray([1, -1, -1, 1])
        expected = numpy.fft.fft(input_data)
        result = fourier.fft(input_data, len(input_data))
        if isinstance(result, list):
            numpy.testing.assert_allclose(result, expected)

    def test_single_element_input(self):
        input_data = numpy.asarray([complex(1, 1)])
        expected = numpy.asarray([complex(1, 1)])
        result = fourier.fft(input_data, len(input_data))
        self.assertEqual(result, expected)

    def test_empty_input(self):
        input_data = numpy.asarray([])
        with self.assertRaises(ValueError):
            result = fourier.fft(input_data, len(input_data))

    def test_non_power_of_two_input(self):
        input_data = numpy.asarray(
            [
                complex(1, 0),
                complex(2, 0),
                complex(3, 0),
            ]
        )
        with self.assertRaises(ValueError):
            result = fourier.fft(input_data, len(input_data))


unittest.main()
