import sys
import unittest


def run_tests():
    try:
        # Import the tests module from the parent directory.
        # This works because 'sstatics' is installed as a package,
        # and the Python path points to the directory before 'sstatics'.
        from sstatics import tests
    except ImportError:
        print("Error: Could not find the tests module.")
        print("Make sure you are running the command in the project's root "
              "directory.")
        sys.exit(1)

    # Load all tests from the module into a test suite.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(tests)

    # Run the tests using a TextTestRunner with low verbosity.
    # verbosity=0 suppresses the individual test output,
    # showing only the summary.
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    # Exit the script with a status code of 1 if any tests failed.
    # This is useful for automation and CI/CD pipelines.
    sys.exit(not result.wasSuccessful())


if __name__ == '__main__':
    # Check if the 'test' argument was provided.
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_tests()
    else:
        # If the command is incorrect, print a helpful usage message.
        print("Unknown command.")
        print("Usage: python -m sstatics test")
