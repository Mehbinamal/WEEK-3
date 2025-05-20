
def sum_of_evens(numbers):
    """
    Calculates the sum of all even numbers in a list.

    Args:
      numbers: A list of numbers.

    Returns:
      The sum of all even numbers in the list.  Returns 0 if the list is empty or contains no even numbers.
    """
    sum_even = 0
    for number in numbers:
        if number % 2 == 0:
            sum_even += number
    return sum_even

# Example usage
my_list = [1, 2, 3, 4, 5, 6]
even_sum = sum_of_evens(my_list)
print(f"The sum of even numbers is: {even_sum}")  # Output: 12

my_list = [1,3,5]
even_sum = sum_of_evens(my_list)
print(f"The sum of even numbers is: {even_sum}") # Output: 0

my_list = []
even_sum = sum_of_evens(my_list)
print(f"The sum of even numbers is: {even_sum}") # Output: 0
