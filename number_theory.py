import random
import math
import torch




def find_primes(lower_limit, upper_limit, d):

    output = []

    limit = int(round(upper_limit))

    boolean = [True] * (limit + 1)

    boolean[0] = boolean[1] = False

    for i in range(2, limit):
        if boolean[i]:
            if i >= lower_limit:
                output.append(i)
                if len(output) == d:
                    return output
            p = i
            for multiple in range(p * p, upper_limit + 1, p):
                boolean[multiple] = False
    output_torch = find_primes_torch(lower_limit, upper_limit, d)
    assert len(output_torch) == len(output)
    zipped = zip(output_torch, output)
    for x,y in zipped:
        assert x == y
    return output


def find_primes_torch(lower_limit, upper_limit, d):
    limit = int(round(upper_limit))
    is_prime = torch.ones(limit + 1, dtype=torch.bool)
    is_prime[:2] = False

    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False

    primes = torch.nonzero(is_prime, as_tuple=True)[0]
    primes = primes[primes >= lower_limit]

    return primes[:d]



def find_omega(p):

    lookup = [True] * p
    lookup[0] = False

    counter = 1
    selected = 0

    while counter != p:
        selected = select_element(lookup, p)
        counter, lookup = filter_multiplicativ(lookup, selected, p, counter)

    return selected



def select_element(lookup, p):

    selected = random.randint(1, p-1)

    while not lookup[selected]:

        selected = random.randint(1, p-1)

    return selected


def filter_multiplicativ(lookup, selected, p, counter):

    lookup[selected] = False
    counter += 1

    current = (selected * selected) % p

    while current != selected:

        if lookup[current]:
            lookup[current] = False
            counter += 1

        current = (current * selected) % p

    return counter, lookup


def is_primitive_root_brute_force(g, p):

    unique_powers = set()

    for k in range(1, p):
        result = pow(g, k, p)
        unique_powers.add(result)

    return len(unique_powers) == p - 1


def estimate_upper_limit(T, n):
    x = T * math.log(T)
    while x / math.log(x) < T:
        x += 1

    return int(x)


def generate_first_t_primes(t):
    """Generates the first t prime numbers using an optimized approach."""
    if not isinstance(t, int):
        try:
            t = int(t)
        except ValueError:
            raise TypeError("The parameter t must be an integer.")
    if t <= 0:
        return []

    # Estimate upper bound using the prime number theorem
    if t < 6:
        upper_bound = 15  # Small heuristic for very small t
    else:
        upper_bound = int(t * (math.log(t) + math.log(math.log(t))))  # PNT estimate

    # Sieve of Eratosthenes
    sieve = [True] * (upper_bound + 1)
    sieve[0] = sieve[1] = False  # 0 and 1 are not primes

    for i in range(2, int(math.sqrt(upper_bound)) + 1):
        if sieve[i]:
            for multiple in range(i * i, upper_bound + 1, i):
                sieve[multiple] = False

    # Collect primes
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]

    # Ensure exactly t primes are returned (may require increasing upper_bound for large t)
    while len(primes) < t:
        upper_bound *= 2
        sieve = [True] * (upper_bound + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(math.sqrt(upper_bound)) + 1):
            if sieve[i]:
                for multiple in range(i * i, upper_bound + 1, i):
                    sieve[multiple] = False

        primes = [i for i, is_prime in enumerate(sieve) if is_prime]

    return primes[:t]


def prime_factors(n):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 2:
        factors.append(n)
    return factors


def is_primitive_root(g, p):
    p_minus_1 = p - 1
    factors = prime_factors(p_minus_1)

    for f in factors:
        if pow(g, p_minus_1 // f, p) == 1:
            return False
    return True


def find_primitive_root(p):
    if p == 2:
        return 1
    for g in range(2, p):
        if is_primitive_root(g, p):
            return g
    return None




def find_primes_for_task(b, N):
    """
    Finds the first x primes greater than b, where x is estimated based on N.
    Uses a PyTorch-based Sieve of Eratosthenes.
    """
    # Compute the number of primes needed (improved heuristic)
    x = max(math.ceil(math.log(N + 1) * 1.2), b)

    # Better upper bound for prime search
    upper_limit = max(100, int(1.5 * x * math.log(x + 2)))

    # Sieve of Eratosthenes using PyTorch
    is_prime = torch.ones(upper_limit + 1, dtype=torch.bool)
    is_prime[:2] = False  # 0 and 1 are not prime

    for i in range(2, int(math.sqrt(upper_limit)) + 1):
        if is_prime[i]:
            is_prime[i * i : upper_limit + 1 : i] = False  # Mark multiples as False

    # Extract prime indices
    primes = torch.nonzero(is_prime, as_tuple=True)[0]

    # Select only primes greater than b
    primes = primes[primes > b]

    # Ensure we return exactly x primes
    if len(primes) < x:
        return find_primes_for_task(b, N * 2)  # Expand search range dynamically

    return primes[:x]
