def calculate_total(price: float, quantity: int) -> float:
    return price * quantity


def main() -> None:
    total = calculate_total(100.0, 3)
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
