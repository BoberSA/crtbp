# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:57:01 2018

@author: stasb
"""

import multiprocessing as mp
from math import ceil
import time
import sys

#-----------------------------------------------------------------------------
# functions for multiprocessing
#-----------------------------------------------------------------------------

def convert_to_d_h_m_s(seconds):
    """Return the tuple of days, hours, minutes and seconds."""

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    return days, hours, minutes, seconds

def async_listener(arg):
    '''
        Asyncronous listener: writes results and runs next jobs
    '''
    writer_fun = arg['fun']
    q = arg['queue']
    jobs = arg['jobs']
    p = arg['p']
    job_count = len(jobs)+p
    s = set(range(job_count))

    print('<pool of %d processes started working on %d jobs>'%(p, job_count))
    sys.stdout.flush()

    while True:
        if q.empty():
            time.sleep(0.1)
        else:
            job = q.get()
            writer_fun(job)
            s.remove(job['id'])
            print('<%.2f%% done>'%((job_count-len(s))*100/job_count))
            sys.stdout.flush()
            if not s:
                print('<all jobs completed>')
                sys.stdout.flush()
                break
            if jobs: # run next job
                next_job = jobs.pop(0)
                next_job['pid']=job['pid']
                mp.Process(target=async_calc, args=(next_job,)).start()
            
def async_calc(arg):
    '''    
        Asyncronous calculation starter
    '''
    t0 = time.clock()
    arg['result']=arg['fun'](arg['job'])
    t = time.clock() - t0
    print('<job time: %.2f s>'%t)
    arg.pop('queue').put(arg)    

def pool_run(p, jobs, calc, writer):
    '''
        Runs simultaneously working processes:
            1 process for sequentially write results to file (calls writer),
            p processes for calculating job (calls calc(job)).
        Processes communicate through shared queue (multiprocessing.Queue).
            
    Parameters
    ----------
        
    p : int
        Number of simultaneously calculation processes.
        
    jobs : iterable
        Some iterable object (for example: list) that consists of
        information for calculation jobs.
        
    calc : function
        Function that make all calculation work.
        
    writer : function
        Function to write result of calculation.
        
        
    Usage example
    -------------
            
        import numpy as np
        
        def calc_job(job):
            # make some calculations
            result = job+0
            
            # sleep some time (not needed in real calculation)
            t = np.random.randint(1, high=5)
            time.sleep(t)
            
            return result
        
        def write_result(job):
            # open file and write result to it
            fname = gen_fname(job)
            with open(fname, 'a') as f:
                r = job.get('result', 'None')
                f.writelines('Result: %s\n'%r)
            print('Result %s was written to file %s'%(job['id'],fname))
        
        def gen_fname(job):
            return 'queue.txt'
        
        if __name__ == '__main__':
            # run asyncronously 8 calculation intensive jobs
            # with pool of 4 workers and 1 writer
            
            p = 4
            jobs = list(range(8))
            pool_run(p, jobs, calc_job, write_result)
        
    '''
    q = mp.Queue()

    calc_jobs = [{'fun':calc, 'queue':q, 'id':i, 'job':j} for i, j in enumerate(jobs)]
    
    first_jobs = calc_jobs[:p]
    t0 = time.clock()
    for pid, j in enumerate(first_jobs):
        j['pid']=pid
        mp.Process(target=async_calc, args=(j,)).start()

    listener_job = {'fun':writer, 'queue':q, 'jobs':calc_jobs[p:], 'p':len(first_jobs)}
    listener = mp.Process(target=async_listener, args=(listener_job,))
    listener.start()
    listener.join()
    dt = time.clock() - t0
    dtfrac = round((dt - ceil(dt))*100)
    print('<time spent: %02d:%02d:%02d:%02d.%02d dd:hh:mm:ss.ss>' % (*convert_to_d_h_m_s(dt), dtfrac))
    print('<time spent: %.2f s>'% dt)
    print('<sequential average time per job: %.2f s>'% (dt/len(jobs)))
    
#-----------------------------------------------------------------------------
# user functions
#-----------------------------------------------------------------------------

def calc_job(job):
    # make some calculations
    result = job+0

    # emulating some work
    i = 1000000000
    while i:
        i-=1
      
    return result

def write_result(job):
    # open file and write result to it
    fname = 'mp_pool_output.txt'
    with open(fname, 'a') as f:
        r = job.get('result', 'None')
        f.writelines('Result: %s\n'%r)
        print('Result %s was written to file %s'%(job['id'],fname))

#-----------------------------------------------------------------------------


if __name__ == '__main__':
    # run asyncronously n calculation intensive jobs
    # with pool of p workers and 1 writer
    n = 8
    p = 8
    jobs = list(range(n))
    pool_run(p, jobs, calc_job, write_result)
