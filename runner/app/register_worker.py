import os, logging, time, threading
import httpx

logger = logging.getLogger(__name__)

class RegisterWorker(threading.Thread):
    def __init__(self):
        self.pipeline = os.environ.get("PIPELINE", "")
        self.model_id = os.environ.get("MODEL_ID","")
        self.orchestrator_url = os.environ.get("ORCHESTRATOR", "")
        self.worker_url = os.environ.get("WORKER_URL","")
        self.orch_secret = os.environ.get("ORCH_SECRET", "")
        self.price = int(os.environ.get("PRICE_PER_UNIT", ""))
        #turn off logging for httpx when no error
        logging.getLogger("httpx").setLevel(logging.ERROR)

    def remove_worker(self):
        model_json = {"pipeline": self.pipeline,
                        "model_id": self.model_id,
                        "url": self.worker_url,
                        "price_per_unit": self.price,
                        "warm": True
                        }
        
        resp = httpx.post(self.orchestrator_url+"/remove-ai-worker", headers={"Credentials":self.orch_secret}, json=[model_json], timeout=30, verify=False)
        if resp.status_code == 200:
            logger.info(f"Worker removed from orchestrator {self.orchestrator_url}")
        else:
            logger.error(f"Worker NOT removed from orchestrator {self.orchestrator_url}, error: {resp.text}")

    def run(self):
        #register worker
        logger.info("register worker task started")
        
        if self.pipeline != "" and self.model_id != "" and self.worker_url != "" and self.orchestrator_url != "" and self.orch_secret != "" and self.price != "":
            model_json = {"pipeline": self.pipeline,
                        "model_id": self.model_id,
                        "url": self.worker_url,
                        "price_per_unit": self.price,
                        "warm": True
                        }

            #run register worker loop to wait for api to be up and confirm registration every 5 minutes
            logger.info("registering worker")
            api_is_running = False
            while True:                
                #let the API start
                while api_is_running == False:
                    resp = httpx.get(self.worker_url+"/health")
                    if resp.status_code < 400:
                        api_is_running = True
                    time.sleep(1)
                #register with the orchestrator
                resp = httpx.post(self.orchestrator_url+"/register-ai-worker", headers={"Credentials":self.orch_secret}, json=[model_json], timeout=30, verify=False)
                if resp.status_code == 200:
                    logger.info(f"Worker registered to orchestrator {self.orchestrator_url}")
                    time.sleep(300)
                elif resp.status_code == 304:
                    logger.info(f"Worker registration confirmed to orchestrator {self.orchestrator_url}")
                    time.sleep(300)
                else:
                    logger.error(f"Worker registration failed to orchestrator {self.orchestrator_url}, error: {resp.text}")
                    time.sleep(1)
                    
        else:
            logger.error("worker registration not possible, need to specify WORKER_URL, ORCHESTRATOR, ORCH_SECRET and PRICE_PER_UNIT environment variables.")
            return