import unittest
import numpy as np

from pluto.core import score
from unittest import mock

class  CoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_score(self):
        # 类型一：检测当天涨停，第二天未涨停
        code = '002750.XSHE'
        bars = np.array([
       ('2022-09-27T00:00:00',  8.55,  9.49, 8.55,  9.49, 19119568., 1.77904128e+08, 6.158),
       ('2022-09-28T00:00:00',  9.35,  9.85, 9.29,  9.36, 24557842., 2.35395644e+08, 6.158),
       ('2022-09-29T00:00:00',  9.26,  9.54, 8.96,  9.03, 14725071., 1.36067616e+08, 6.158),
       ('2022-09-30T00:00:00',  9.05,  9.93, 9.  ,  9.93, 11580068., 1.12574827e+08, 6.158)],
      dtype=[('frame', '<M8[s]'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'), 
      ('close', '<f4'), ('volume', '<f8'), ('amount', '<f8'), ('factor', '<f4')])
        with mock.patch('omicron.models.stock.Stock.trade_price_limit_flags', side_effect=[(True, False), (True, False)]):
            actual = await score(bars, code)
            exp = ([0.00106952, -0.0342246 ,  0.06203209],
                    [0.062032085561497335],
                    [-0.03422459893048131])
            for i in range(3):
                np.testing.assert_array_almost_equal(actual[i],exp[i],decimal=3)

        # 类型二： 检测当天涨停，第二天开盘也涨停
        code = '002380.XSHE'
        bars = np.array([
       ('2022-06-07T00:00:00', 10.12, 11.04,  9.92, 11.04,  7192401., 7.66028270e+07, 3.342714),
       ('2022-06-08T00:00:00', 12.1 , 12.14, 12.1 , 12.14,  4939200., 5.99058860e+07, 3.342714),
       ('2022-06-09T00:00:00', 13.35, 13.35, 13.01, 13.35,  8613404., 1.14910770e+08, 3.342714),
       ('2022-06-10T00:00:00', 14.17, 14.69, 13.97, 14.69, 16849817., 2.43503344e+08, 3.342714),
       ('2022-06-13T00:00:00', 15.  , 16.16, 14.81, 16.16, 25108534., 3.92079273e+08, 3.342714)],
      dtype=[('frame', '<M8[s]'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'), ('close', '<f4'), 
        ('volume', '<f8'), ('amount', '<f8'), ('factor', '<f4')])
        with mock.patch('omicron.models.stock.Stock.trade_price_limit_flags', return_value=([True], [False])):
            actual = await score(bars, code)
            exp = ([0.00330579, 0.10330579, 0.21404959, 0.33553719],
                    [0.33553719008264465],
                    [])
            for i in range(3):
                np.testing.assert_array_almost_equal(actual[i],exp[i],decimal=3)

        # 类型三： 检测当天未涨停
        code = '002380.XSHE'
        bars = np.array([
       ('2022-07-13T00:00:00', 12.12    , 12.63     , 12.05    , 12.43    ,  7019231.        , 8.68321520e+07, 3.353773),
       ('2022-07-14T00:00:00', 12.52    , 13.15     , 12.46    , 12.65    , 11040618.        , 1.41468480e+08, 3.353773),
       ('2022-07-15T00:00:00', 12.49    , 12.56     , 12.2     , 12.2     ,  8024077.        , 9.89049220e+07, 3.353773),
       ('2022-07-18T00:00:00', 12.58    , 12.96     , 12.46    , 12.8     , 10895999.        , 1.39033790e+08, 3.353773),
       ('2022-07-19T00:00:00', 12.87    , 13.34     , 12.65    , 13.23    , 16645821.        , 2.17165421e+08, 3.353773)],
      dtype=[('frame', '<M8[s]'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'), ('close', '<f4'), ('volume', '<f8'), 
      ('amount', '<f8'), ('factor', '<f4')])
        with mock.patch('omicron.models.stock.Stock.trade_price_limit_flags', return_value=([False], [False])):
            actual = await score(bars, code)
            exp = ([ 0.01769912, -0.01850362,  0.02976669,  0.06436042],
                    [0.06436041834271929],
                    [-0.018503620273531814])
            for i in range(3):
                np.testing.assert_array_almost_equal(actual[i],exp[i],decimal=3)


