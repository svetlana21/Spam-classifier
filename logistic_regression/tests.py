'''
Тесты для функций normalization и division.
'''

import unittest
import numpy as np
from numpy import testing
try:
    from unittest import mock
except ImportError:
    import mock

from spam_lg import LogReg

class TestLogReg(unittest.TestCase):

    def setUp(self):
        self.test_logreg = LogReg()

    @mock.patch('random.shuffle')
    def test_division(self, mock_shuffle):
        """
        Тестирование функции division
        """        
        test_data = [[2.4, 4, 6, 9, 8.9, 0], [1.2, 4, 3, 6, 7.1, 1], [1.2, 2, 3, 4, 5.5, 1], [4.6, 6, 7, 9, 0.5, 0], [2.3, 5, 6, 2, 6.7, 1], 
                     [1.2, 2, 3, 4, 5.5, 1], [4.6, 6, 7, 9, 0.5, 0], [2.3, 5, 6, 2, 6.7, 1], [2.4, 4, 6, 9, 8.9, 0], [1.2, 4, 3, 6, 7.1, 1]]
        mock_shuffle.return_value = test_data
        true_result = (np.array([[2.4, 4, 6, 9, 8.9], [1.2, 4, 3, 6, 7.1], [1.2, 2, 3, 4, 5.5], [4.6, 6, 7, 9, 0.5], [2.3, 5, 6, 2, 6.7], 
                     [1.2, 2, 3, 4, 5.5], [4.6, 6, 7, 9, 0.5], [2.3, 5, 6, 2, 6.7]]),
                       np.array([0, 1, 1, 0, 1, 1, 0, 1]),
                       np.array([[2.4, 4, 6, 9, 8.9], [1.2, 4, 3, 6, 7.1]]),
                       np.array([0, 1]))
        fact_result = self.test_logreg.division(test_data)
        
        np.testing.assert_array_equal(true_result[0], fact_result[0])
        np.testing.assert_array_equal(true_result[1], fact_result[1])
        np.testing.assert_array_equal(true_result[2], fact_result[2])
        np.testing.assert_array_equal(true_result[3], fact_result[3])
        
    def test_normalization_without_coefs(self):
        """
        Тестирование функции normalization без передачи ей коэффициентов
        """
        test_data = np.array([[1.2, 2], [4, 8]])
        true_result = (np.array([[0.29, 0.24], [0.96, 0.97]]), [4.18, 8.25])
        fact_result = self.test_logreg.normalization(test_data)
        rounded_coefs = [round(el, 2) for el in fact_result[1]]

        np.testing.assert_array_equal(np.around(fact_result[0], decimals=2), true_result[0])
        self.assertCountEqual(true_result[1], rounded_coefs)
        
    def test_normalization_with_coefs(self):
        """
        Тестирование функции normalization с передачей ей коэффициентов
        """
        test_data = np.array([[1.2, 2], [4, 8]])
        coefs = [4.18, 8.25]
        true_result = (np.array([[0.29, 0.24], [0.96, 0.97]]), [4.18, 8.25])
        fact_result = self.test_logreg.normalization(test_data, coefs)
        rounded_coefs = [round(el, 2) for el in fact_result[1]]

        np.testing.assert_array_equal(np.around(fact_result[0], decimals=2), true_result[0])
        self.assertCountEqual(true_result[1], rounded_coefs)

if __name__ == '__main__':
    unittest.main()
