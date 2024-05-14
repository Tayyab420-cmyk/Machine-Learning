menu = {
    1: {"name": 'espresso',
        "price": 1.99},
    2: {"name": 'coffee', 
        "price": 2.50},
    3: {"name": 'cake', 
        "price": 2.79},
    4: {"name": 'soup', 
        "price": 4.50},
    5: {"name": 'sandwich',
        "price": 4.99}
}

def calculate_subtotal(order):
    """Calculates the subtotal of an order."""
    print('Calculating bill subtotal...')
    subtotal = sum(item['price'] for item in order)
    return subtotal

def calculate_tax(subtotal):
    """Calculates the tax of an order."""
    print('Calculating tax from subtotal...')
    tax = round(subtotal * 0.15, 2)
    return tax

def summarize_order(order):
    """Summarizes the order."""
    print('Summarizing order...')
    names = [item['name'] for item in order]
    subtotal = calculate_subtotal(order)
    tax = calculate_tax(subtotal)
    total = round(subtotal + tax, 2)
    return names, total

# Test the functions
sample_order = [
    {"name": 'espresso', "price": 1.99},
    {"name": 'coffee', "price": 2.50},
    {"name": 'cake', "price": 2.79}
]

names, total = summarize_order(sample_order)
print("Items ordered:", names)
print("Total amount (including tax):", total)
