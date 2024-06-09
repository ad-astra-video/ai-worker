import os, logging, time, threading
import httpx

logger = logging.getLogger(__name__)

class RegisterWorker(threading.Thread):
    def run(self):
        #register worker
        logger.info("register worker task started")
        pipeline = os.environ.get("PIPELINE", "")
        model_id = os.environ.get("MODEL_ID","")
        orchestrator_url = os.environ.get("ORCHESTRATOR", "")
        worker_url = os.environ.get("WORKER_URL","")
        orch_secret = os.environ.get("ORCH_SECRET", "")
        price = int(os.environ.get("PRICE_PER_UNIT", ""))
        if pipeline != "" and model_id != "" and worker_url != "" and orchestrator_url != "" and orch_secret != "" and price != "":
            model_json = {"pipeline": pipeline,
                        "model_id": model_id,
                        "url": worker_url,
                        "price_per_unit": price,
                        "warm": True
                        }
            
            #turn off logging for httpx when no error
            logging.getLogger("httpx").setLevel(logging.ERROR)

            #run register worker loop to wait for api to be up and confirm registration every 5 minutes
            logger.info("registering worker")
            api_is_running = False
            while True:                
                #let the API start
                while api_is_running == False:
                    resp = httpx.get(worker_url+"/health")
                    if resp.status_code < 400:
                        api_is_running = True
                    time.sleep(1)
                #register with the orchestrator
                resp = httpx.post(orchestrator_url+"/register-ai-worker", headers={"Credentials":orch_secret}, json=[model_json], timeout=30, verify=False)
                if resp.status_code == 200:
                    logger.info(f"Worker registered to orchestrator {orchestrator_url}")
                    time.sleep(300)
                elif resp.status_code == 304:
                    logger.info(f"Worker registration confirmed to orchestrator {orchestrator_url}")
                    time.sleep(300)
                else:
                    logger.error(f"Worker registration failed to orchestrator {orchestrator_url}, error: {resp.text}")
                    time.sleep(1)
                    
        else:
            logger.error("worker registration not possible, need to specify WORKER_URL, ORCHESTRATOR, ORCH_SECRET and PRICE_PER_UNIT environment variables.")
            return