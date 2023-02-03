import sys
import os
sys.path.insert(1, os.path.expanduser('~/Documents/hyp3-sdk/src'))
import hyp3_sdk as sdk

hyp3 = sdk.HyP3()
batch = hyp3.find_jobs().filter_jobs(succeeded = False, failed = False)
print(batch._count_statuses())
hyp3.watch(batch)