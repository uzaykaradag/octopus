import pickle
from octopus.core import predict

def test_picklable():
    try:
        # Attempt to pickle the trace_elm function
        pickled_func = pickle.dumps(predict.trace_elm)
        # Attempt to unpickle it
        unpickled_func = pickle.loads(pickled_func)
        print("trace_elm is picklable.")
        return True
    except Exception as e:
        print(f"trace_elm is not picklable. Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_picklable()