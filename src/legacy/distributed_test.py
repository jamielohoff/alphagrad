import jax
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--pid", type=int,
                    default=0, help="id of this process in the Jax virtual cluster.")

args = parser.parse_args()

jax.distributed.initialize(coordinator_address="134.94.166.3:1234",
                           	num_processes=2,
                            process_id=args.pid) # On GPU, see above for the necessary arguments.
print(jax.device_count())  # total number of accelerator devices in the cluster
print(jax.local_device_count())  # number of accelerator devices attached to this host

print(jax.devices())
print(jax.local_devices())
# The psum is performed over all mapped devices across the pod slice
xs = jax.numpy.ones(jax.local_device_count())
res = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i', devices=jax.devices())(xs)

if args.pid == 0:
    print(res)
