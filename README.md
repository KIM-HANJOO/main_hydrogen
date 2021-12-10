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

## facing problems

at facility '판매 및 숙박'

'판매' and '숙박' are set apart in model 1,
and are merged then clustered in model 3

when generating the profiles with the whole models, 
need to decide rather to mix or not
