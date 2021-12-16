# main_hydrogen

## facing errors

from '\module\3_profile_generator\temp_profile_generator.py'

in generating profiles, with very rare chances with line
'weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)'

ValueError : scale < 0

error rises.
scale means the spread (or width) of Standard deviation of the distribution,
rises with a small chance(smaller than 1%).
with plots of results, errors cannot be found yet

1. with adding 's = np.random.normal(mu, np.abs(sigma),1000)', 
able to check the sign of the parameter 's'


---
problem arises when the scale value (st_week) is shown to be negative.
st_week is made at previous lines, with random sampling code 'beta.rvs(alpha, beta, loc, scale, num)'

adjusted code by adding while-loop until st_week is not negative.
with the alpha, beta, loc, scale values given, have to aware that there are chances for the random sampled values to be negative.



2. in making model2, with a little chance -
sheet size can go beyond the limit size.

## facing problems

at facility '판매 및 숙박'

'판매' and '숙박' are set apart in model 1,
and are merged then clustered in model 3

when generating the profiles with the whole models, 
need to decide rather to mix or not
