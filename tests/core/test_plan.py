import unittest

import numpy as np
from omicron.talib import moving_average

from pluto.core.plan import ma_support_prices, magic_numbers, predict_next_ma


class PlanTest(unittest.TestCase):
    def test_magic_numbers(self):
        actual = magic_numbers(9.3, 9.3, 9.1)
        exp = [8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.3, 9.1]
        np.testing.assert_array_almost_equal(actual, exp, 2)

    def test_ma_support_prices(self):
        # 朗迪集团 2022-10-25
        # fmt: off
        close = np.array([13.609065 , 13.18348  , 13.231842 , 12.990032 , 13.135118 ,
       13.289876 , 13.2608595, 13.038394 , 13.241514 , 13.193152 ,
       13.560704 , 13.280204 , 13.6671   , 13.483324 , 13.212497 ,
       13.173807 , 13.202825 , 12.970687 , 12.593464 , 12.516084 ,
       12.641826 , 12.603136 , 12.622481 , 12.748222 , 12.7578945,
       12.719205 , 12.825602 , 12.951343 , 13.154463 , 13.376928 ,
       13.531686 , 13.957273 , 13.676773 , 13.937927 , 13.937927 ,
       14.053996 , 14.363513 , 14.450564 , 14.450564 , 14.382857 ,
       14.024979 , 14.005633 , 14.218427 , 14.43122  , 14.315151 ,
       14.324823 , 14.073341 , 13.918583 , 13.90891  , 14.005633 ,
       13.928254 , 14.044324 , 15.446822 , 14.769754 , 14.769754 ,
       14.556961 , 14.605323 , 14.943856 , 14.663357 , 13.879893 ,
       13.483324 , 13.522014 , 13.589721 , 13.599394 , 13.560704 ,
       13.725134 , 13.879893 , 13.618738 , 13.841204 , 13.609065 ,
       13.70579  , 14.024979 , 14.083014 , 14.276462 , 14.170064 ,
       14.199082 , 13.976617 , 14.199082 , 13.6671   , 14.228099 ,
       13.986289 , 13.202825 , 13.299549 , 12.699861 , 12.864291 ,
       13.01905  , 13.202825 , 13.396274 , 13.309221 , 13.251186 ,
       13.280204 , 13.676773 , 15.040581 , 15.175994 , 15.756339 ,
       16.220613 , 15.76601  , 14.818115 , 15.059925 , 14.972874 ,
       14.67303  , 14.808443 , 14.721392 , 14.702046 , 14.644012 ,
       14.402203 , 13.850875 , 13.580049 , 13.686444 , 14.063668 ,
       13.734807 , 12.999704 , 13.057739 , 13.241514 , 13.3866005,
       13.638083 , 13.444634 , 13.347911 , 13.280204 , 13.318894 ,
       12.961015 , 12.806256 , 12.893308 , 12.893308 , 12.844946 ,
       12.961015 , 12.593464 , 12.370998 , 11.984103 , 12.3129635,
       12.100172 , 12.148534 , 11.906724 , 12.051809 , 12.148534 ,
       12.25493  , 11.693931 , 11.674585 , 10.833087 , 10.697674 ,
       10.968501 , 10.56226  , 10.862103 , 10.920138 , 10.871777 ,
       10.978173 , 11.016863 , 11.007191 , 11.04588  , 11.200638 ,
       11.132932 , 11.219983 , 11.28769  , 11.384415 , 11.500483 ,
       11.645569 , 11.210311 , 11.442448 , 11.616551 , 11.519827 ,
       11.374742 , 11.5295   , 11.5295   , 11.606879 , 11.722948 ,
       11.587534 , 11.548844 , 11.413431 , 11.519827 , 11.413431 ,
       11.413431 , 11.423104 , 11.510155 , 11.616551 , 11.693931 ,
       11.829344 , 11.809999 , 11.77     , 12.08     , 12.06     ,
       12.3      , 11.99     , 12.       , 12.04     , 12.1      ,
       11.95     , 11.87     , 11.79     , 11.97     , 11.88     ,
       12.05     , 12.39     , 12.61     , 12.09     , 12.42     ,
       12.43     , 12.53     , 12.52     , 12.45     , 12.38     ,
       12.42     , 12.52     , 12.75     , 12.81     , 12.93     ,
       12.43     , 12.32     , 12.41     , 13.65     , 15.02     ,
       14.79     , 14.77     , 15.35     , 14.95     , 14.93     ,
       14.74     , 15.19     , 14.8      , 14.17     , 13.92     ,
       13.8      , 13.12     , 12.86     , 12.82     , 13.05     ,
       13.3      , 13.07     , 13.05     , 13.65     , 13.54     ,
       13.54     , 13.55     , 13.28     , 13.39     , 13.26     ,
       13.26     , 13.03     , 12.7      , 12.78     , 13.46     ,
       13.5      , 13.64     , 13.36     , 12.93     , 13.47     ,
       14.36     , 13.77     , 13.35     , 13.52     , 13.1      ,
       14.41     , 14.9      , 14.68     , 14.83     , 14.55     ,
       14.13     , 14.32     , 14.35     , 14.63     , 14.02     ],
      dtype=np.float32)

        # fmt: on
        mas = {}
        for win in (5, 10, 20, 30, 60, 120, 250):
            mas[win] = moving_average(close, win)
            if win == 30:
                print(np.round(mas[win][-7:], 2))

        actual = ma_support_prices(mas, close[-1])
        exp = {5: None, 10: None, 20: -1, 30: 13.76, 60: 13.72, 120: 12.7, 250: 13.16}
        self.assertDictEqual(actual, exp)
        for win in (5, 10, 20, 30, 60, 120, 250):
            if actual.get(win):
                gap = actual.get(win) / close[-1]
                self.assertTrue(gap > 0.9 or gap < 0)

    def test_predict_next_ma(self):
        # 朗迪集团 2022-10-25
        # fmt: off
        close = np.array([13.609065 , 13.18348  , 13.231842 , 12.990032 , 13.135118 ,
       13.289876 , 13.2608595, 13.038394 , 13.241514 , 13.193152 ,
       13.560704 , 13.280204 , 13.6671   , 13.483324 , 13.212497 ,
       13.173807 , 13.202825 , 12.970687 , 12.593464 , 12.516084 ,
       12.641826 , 12.603136 , 12.622481 , 12.748222 , 12.7578945,
       12.719205 , 12.825602 , 12.951343 , 13.154463 , 13.376928 ,
       13.531686 , 13.957273 , 13.676773 , 13.937927 , 13.937927 ,
       14.053996 , 14.363513 , 14.450564 , 14.450564 , 14.382857 ,
       14.024979 , 14.005633 , 14.218427 , 14.43122  , 14.315151 ,
       14.324823 , 14.073341 , 13.918583 , 13.90891  , 14.005633 ,
       13.928254 , 14.044324 , 15.446822 , 14.769754 , 14.769754 ,
       14.556961 , 14.605323 , 14.943856 , 14.663357 , 13.879893 ,
       13.483324 , 13.522014 , 13.589721 , 13.599394 , 13.560704 ,
       13.725134 , 13.879893 , 13.618738 , 13.841204 , 13.609065 ,
       13.70579  , 14.024979 , 14.083014 , 14.276462 , 14.170064 ,
       14.199082 , 13.976617 , 14.199082 , 13.6671   , 14.228099 ,
       13.986289 , 13.202825 , 13.299549 , 12.699861 , 12.864291 ,
       13.01905  , 13.202825 , 13.396274 , 13.309221 , 13.251186 ,
       13.280204 , 13.676773 , 15.040581 , 15.175994 , 15.756339 ,
       16.220613 , 15.76601  , 14.818115 , 15.059925 , 14.972874 ,
       14.67303  , 14.808443 , 14.721392 , 14.702046 , 14.644012 ,
       14.402203 , 13.850875 , 13.580049 , 13.686444 , 14.063668 ,
       13.734807 , 12.999704 , 13.057739 , 13.241514 , 13.3866005,
       13.638083 , 13.444634 , 13.347911 , 13.280204 , 13.318894 ,
       12.961015 , 12.806256 , 12.893308 , 12.893308 , 12.844946 ,
       12.961015 , 12.593464 , 12.370998 , 11.984103 , 12.3129635,
       12.100172 , 12.148534 , 11.906724 , 12.051809 , 12.148534 ,
       12.25493  , 11.693931 , 11.674585 , 10.833087 , 10.697674 ,
       10.968501 , 10.56226  , 10.862103 , 10.920138 , 10.871777 ,
       10.978173 , 11.016863 , 11.007191 , 11.04588  , 11.200638 ,
       11.132932 , 11.219983 , 11.28769  , 11.384415 , 11.500483 ,
       11.645569 , 11.210311 , 11.442448 , 11.616551 , 11.519827 ,
       11.374742 , 11.5295   , 11.5295   , 11.606879 , 11.722948 ,
       11.587534 , 11.548844 , 11.413431 , 11.519827 , 11.413431 ,
       11.413431 , 11.423104 , 11.510155 , 11.616551 , 11.693931 ,
       11.829344 , 11.809999 , 11.77     , 12.08     , 12.06     ,
       12.3      , 11.99     , 12.       , 12.04     , 12.1      ,
       11.95     , 11.87     , 11.79     , 11.97     , 11.88     ,
       12.05     , 12.39     , 12.61     , 12.09     , 12.42     ,
       12.43     , 12.53     , 12.52     , 12.45     , 12.38     ,
       12.42     , 12.52     , 12.75     , 12.81     , 12.93     ,
       12.43     , 12.32     , 12.41     , 13.65     , 15.02     ,
       14.79     , 14.77     , 15.35     , 14.95     , 14.93     ,
       14.74     , 15.19     , 14.8      , 14.17     , 13.92     ,
       13.8      , 13.12     , 12.86     , 12.82     , 13.05     ,
       13.3      , 13.07     , 13.05     , 13.65     , 13.54     ,
       13.54     , 13.55     , 13.28     , 13.39     , 13.26     ,
       13.26     , 13.03     , 12.7      , 12.78     , 13.46     ,
       13.5      , 13.64     , 13.36     , 12.93     , 13.47     ,
       14.36     , 13.77     , 13.35     , 13.52     , 13.1      ,
       14.41     , 14.9      , 14.68     , 14.83     , 14.55     ,
       14.13     , 14.32     , 14.35     , 14.63     , 14.02     ],
      dtype=np.float32)

        # fmt: on
        exp = {
            5: None,
            10: 14.61,
            20: 14.08,
            30: 13.76,
            60: 13.72,
            120: 12.7,
            250: 13.16,
        }

        for win in (5, 10, 20, 30, 60, 120, 250):
            ma = moving_average(close, win)
            if win == 30:
                print(np.round(ma[-7:], 2))
            actual = predict_next_ma(ma, win)
            if exp.get(win) is None:
                self.assertIsNone(actual[0])
            else:
                self.assertAlmostEqual(actual[0], exp.get(win), 2)
