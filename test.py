from unittest.mock import patch

import pytest


@pytest.mark.unittest
class TestRunTest:
    def test_main(self):
        with patch('sys.argv', ['demo.py']):
            from demo import main
            main()

    def test_main_2(self):
        with patch('sys.argv', ['demo.py', '--psize', '256']):
            from demo import main
            main()
