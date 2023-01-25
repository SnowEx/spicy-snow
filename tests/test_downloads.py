import unittest

from shapely.geometry import box
import pandas as pd
import xarray

import sys
sys.path.append('../c-snow')
from download.sentinel1 import s1_img_search, download_s1_imgs

class TestSentinel1Search(unittest.TestCase):
    """
    Test functionality of searching sentinel-1 granule names from dates and geometry.
    """

    area = box(-115, 46, -115.5, 46.5)
    dates = ('2019-12-30','2020-01-02')
    search_result = s1_img_search(area, dates)

    def test_search_return_type(self, search_result = search_result):
        """
        Test return type of Sentinel-1 search
        """
        self.assertEqual(type(search_result), pd.DataFrame, "Seach results should return DataFrame")
    
    def test_search_return_number(self, search_result = search_result):
        """
        Test if sentinel-1 search returns expected number of results
        """
        # TODO Add in expected number of Sentinel-1 images
        self.assertEqual(len(search_result), XXX, "Seach results should contain XXX images")
    
    def test_search_return_values(self, search_result = search_result):
        """
        Test if sentinel-1 search returns correct images in dataframe
        """
        # TODO Add in expected results
        expected_result = pd.DataFrame()
        self.assertEqual(search_result, expected_result, "Search result should match.")
    
    def test_search_exceptions(self, area = area, dates = dates):
        """
        Test thrown exceptions/errors in sentinel 1 image searches.
        """
        # Incorrect number/type of inputs
        self.assertRaises(ValueError, s1_img_search(area, ('2019-12-30')))
        self.assertRaises(ValueError, s1_img_search(area, (1)))
        self.assertRaises(ValueError, s1_img_search(area, ()))
        self.assertRaises(ValueError, s1_img_search([-115, 46, -115.5, 46.5], dates))
        self.assertRaises(ValueError, s1_img_search([], dates))
        self.assertRaises(ValueError, s1_img_search(['-115', '46', '-115.5', '46.5'], dates))

        # TODO Logical errors in inputs
        # dates are reverse
        self.assertRaises(XXXX_Error, s1_img_search(area, pd.timedelta_range('2021-01-02','2020-01-02')))
        # dates are in future or too far in past
        self.assertRaises(XXXX_Error, s1_img_search(area, pd.timedelta_range('2030-01-02','2032-01-02')))
        self.assertRaises(XXXX_Error, s1_img_search(area, pd.timedelta_range('1999-01-02','2000-01-02')))
        # geometry outside of acceptable bounds or in non-geographic coordinates
        self.assertRaises(XXXX_Error, s1_img_search(bbox(-115, -46, -115.5, -46.5), dates))
        self.assertRaises(XXXX_Error, s1_img_search(bbox(300000, 50000, 300000, 55000), dates))

class TestSentinel1Download(unittest.TestCase):
    """
    Test functionality of downloading sentinel-1 images from url search results.
    """

    # TODO Create expected result dataframe (same as above)
    urls = pd.DataFrame()
    s1_dataset = download_s1_imgs(urls)

    def test_download_return_type(self, dataset = s1_dataset):
        """
        Test return type of Sentinel-1 download
        """
        self.assertEqual(type(dataset), xarray.Dataset, "Download results should return xarray Dataset")
    
    def test_download_return_number(self, dataset = s1_dataset):
        """
        Test if sentinel-1 download returns expected number of results
        """
        # TODO Add in expected number of Sentinel-1 images
        self.assertEqual(len(dataset), XXX, "Download dataset should contain XXX images")
    
    def test_download_return_values(self, dataset = s1_dataset):
        """
        Test if sentinel-1 download returns correct images in dataframe
        """
        # TODO Add in expected results
        expected_result = xarray.Dataset()
        self.assertEqual(dataset, expected_result, "Download result should match.")
    
    def test_download_exceptions(self, urls = urls):
        """
        Test thrown exceptions/errors in sentinel 1 image downloads
        """
        # Incorrect number/type of inputs
        self.assertRaises(ValueError, download_s1_imgs(1))
        self.assertRaises(ValueError, download_s1_imgs('1'))
        self.assertRaises(ValueError, download_s1_imgs(True))

        # TODO Logical errors in inputs
        # urls are invalid
        incorrect_urls = pd.DataFrame()
        self.assertRaises(XXXX_Error, download_s1_imgs(incorrect_urls), "Incorrect urls should return error")
        empty_urls = pd.DataFrame()
        self.assertRaises(XXXX_Error, download_s1_imgs(empty_urls), "Empty dataset should return error")

if __name__ == '__main__':
    unittest.main()