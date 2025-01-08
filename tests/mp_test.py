import multiprocessing

class PersistentStatePool:
    def __init__(self, num_processes, states):
        """
        Initialize the class with a pool and unique states for each process.

        Args:
            num_processes (int): Number of processes.
            states (list): List of unique state values for each process.
        """
        self.num_processes = num_processes
        self.states = states
        self.pool = multiprocessing.Pool(
            processes=self.num_processes,
            initializer=self._init_worker,
            initargs=(None,)  # Placeholder for state initialization
        )
        self._assign_states_to_workers()

    @staticmethod
    def _init_worker(state):
        """
        Worker initializer to set up the global state.
        """
        global process_state
        process_state = state

    def _assign_states_to_workers(self):
        """
        Assign unique states to each worker process.
        """
        for state in self.states:
            self.pool.apply_async(self._init_worker, args=(state,))

    @staticmethod
    def worker_task(x):
        """
        Worker function to process data using the unique process state.
        """
        global process_state
        return x, f"Input: {x}, Process State: {process_state}, Result: {x + process_state}"

    def process_data_with_order(self, data):
        """
        Process a list of data using the worker pool and return results as an ordered dictionary.

        Args:
            data (list): List of inputs to process.

        Returns:
            dict: Results with deterministic output ordering.
        """
        # Use apply_async for manual ordering
        results = {}
        jobs = [self.pool.apply_async(self.worker_task, args=(x,)) for x in data]

        # Collect results and maintain order
        for job in jobs:
            key, result = job.get()
            results[key] = result

        # Ensure deterministic ordering
        return dict(sorted(results.items()))

    def close(self):
        """
        Close the pool after all tasks are completed.
        """
        self.pool.close()
        self.pool.join()


if __name__ == "__main__":
    # Number of processes
    num_processes = 4

    # Unique states for each process
    unique_states = [10, 20, 30, 40]

    # Data to process
    inputs = [8, 3, 1, 6, 4, 7, 2, 5]

    # Create the PersistentStatePool instance
    pool_instance = PersistentStatePool(num_processes, unique_states)

    # Process data with deterministic output ordering
    ordered_results = pool_instance.process_data_with_order(inputs)

    # Print results
    print("Ordered Results:")
    for key, result in ordered_results.items():
        print(f"Input {key}: {result}")

    # Clean up the pool
    pool_instance.close()