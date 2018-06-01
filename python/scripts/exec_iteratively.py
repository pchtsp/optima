import scripts.exec as exec
import os
import datetime as dt
import time

if __name__ == "__main__":
    import package.params as params

    for period in [30, 50, 60]:
        # we don't care about the original params object.
        # so we do not copy it.
        params.OPTIONS['end_pos'] = period
        # params.OPTIONS['solver'] = solver
        params.OPTIONS['path'] = os.path.join(
            params.PATHS['experiments'],
            dt.datetime.now().strftime("%Y%m%d%H%M")
        ) + '/'

        exec.config_and_solve(params)
        time.sleep(60)