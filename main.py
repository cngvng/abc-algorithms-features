from helpers.models import run_compare, run_cart, run_cart_abc

path = '/Users/cngvng/Desktop/abc-algorithms-features/data/german.txt'

run_compare(path)
run_cart(path, 0.3)
run_cart_abc(path, 0.3)