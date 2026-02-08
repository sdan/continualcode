from main import add, calculate, divide, main, multiply, subtract


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(3, 4) == 12


def test_divide():
    assert divide(5, 2) == 2.5


def test_divide_zero():
    try:
        divide(1, 0)
        assert False, "expected ValueError"
    except ValueError as err:
        assert "division by zero" in str(err)


def test_calculate_dispatch():
    assert calculate("add", 1.5, 2.5) == 4.0


def test_cli_preserves_decimal_precision(capsys):
    code = main(["--op", "divide", "--a", "5.0", "--b", "2.0"])
    out = capsys.readouterr().out.strip()
    assert code == 0
    assert out == "2.5"
