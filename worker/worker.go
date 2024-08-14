package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strconv"
	"sync"
	"time"
)

// EnvValue unmarshals JSON booleans as strings for compatibility with env variables.
type EnvValue string

// UnmarshalJSON converts JSON booleans to strings for EnvValue.
func (sb *EnvValue) UnmarshalJSON(b []byte) error {
	var boolVal bool
	err := json.Unmarshal(b, &boolVal)
	if err == nil {
		*sb = EnvValue(strconv.FormatBool(boolVal))
		return nil
	}

	var strVal string
	err = json.Unmarshal(b, &strVal)
	if err == nil {
		*sb = EnvValue(strVal)
	}

	return err
}

// String returns the string representation of the EnvValue.
func (sb EnvValue) String() string {
	return string(sb)
}

// OptimizationFlags is a map of optimization flags to be passed to the pipeline.
type OptimizationFlags map[string]EnvValue

type Worker struct {
	manager             *DockerManager
	noManagedContainers bool

	externalContainers map[string]*RunnerContainer
	mu                 *sync.Mutex
}

func NewWorker(containerImageID string, gpus []string, modelDir string, noManagedContainers bool) (*Worker, error) {
	manager, err := NewDockerManager(containerImageID, gpus, modelDir)
	if err != nil {
		return nil, err
	}

	return &Worker{
		manager:             manager,
		noManagedContainers: noManagedContainers,
		externalContainers:  make(map[string]*RunnerContainer),
		mu:                  &sync.Mutex{},
	}, nil
}

func (w *Worker) TextToImage(ctx context.Context, req TextToImageJSONRequestBody) (*ImageResponse, error) {
	c, err := w.borrowContainer(ctx, "text-to-image", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	resp, err := c.Client.TextToImageWithResponse(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 422", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 400", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 400")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 500", slog.String("err", string(val)))
		c.hadRunError = true
		return nil, errors.New("text-to-image container returned 500")
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToImage(ctx context.Context, req ImageToImageMultipartRequestBody) (*ImageResponse, error) {
	c, err := w.borrowContainer(ctx, "image-to-image", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewImageToImageMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.ImageToImageWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 422", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 400", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 400")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 500", slog.String("err", string(val)))
		c.hadRunError = true
		return nil, errors.New("image-to-image container returned 500")
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToVideo(ctx context.Context, req ImageToVideoMultipartRequestBody) (*VideoResponse, error) {
	c, err := w.borrowContainer(ctx, "image-to-video", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewImageToVideoMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.ImageToVideoWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 422", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 400", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 400")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 500", slog.String("err", string(val)))
		c.hadRunError = true
		return nil, errors.New("image-to-video container returned 500")
	}

	if resp.JSON200 == nil {
		slog.Error("image-to-video container returned no content")
		return nil, errors.New("image-to-video container returned no content")
	}

	return resp.JSON200, nil
}

func (w *Worker) Upscale(ctx context.Context, req UpscaleMultipartRequestBody) (*ImageResponse, error) {
	c, err := w.borrowContainer(ctx, "upscale", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewUpscaleMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.UpscaleWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 422", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 400", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 400")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 500", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 500")
	}

	return resp.JSON200, nil
}

func (w *Worker) AudioToText(ctx context.Context, req AudioToTextMultipartRequestBody) (*TextResponse, error) {
	c, err := w.borrowContainer(ctx, "audio-to-text", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewAudioToTextMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.AudioToTextWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 422", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 400", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 400")
	}

	if resp.JSON413 != nil {
		msg := "audio-to-text container returned 413 file too large; max file size is 50MB"
		slog.Error("audio-to-text container returned 413", slog.String("err", string(msg)))
		return nil, errors.New(msg)
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 500", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 500")
	}

	return resp.JSON200, nil
}

func (w *Worker) Warm(ctx context.Context, pipeline string, modelID string, endpoint RunnerEndpoint, optimizationFlags OptimizationFlags) error {
	if endpoint.URL == "" {
		return w.manager.Warm(ctx, pipeline, modelID, optimizationFlags)
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	cfg := RunnerContainerConfig{
		Type:             External,
		Pipeline:         pipeline,
		ModelID:          modelID,
		Endpoint:         endpoint,
		containerTimeout: externalContainerTimeout,
	}
	rc, err := NewRunnerContainer(ctx, cfg, endpoint.URL)
	if err != nil {
		return err
	}

	name := dockerContainerName(pipeline, modelID)
	if endpoint.URL != "" {
		name = cfg.Endpoint.URL
		slog.Info("name of container: ", slog.String("url", cfg.Endpoint.URL))
	}
	slog.Info("Starting external container", slog.String("name", name), slog.String("pipeline", pipeline), slog.String("modelID", modelID))
	w.externalContainers[name] = rc

	return nil
}

func (w *Worker) Stop(ctx context.Context) error {
	if err := w.manager.Stop(ctx); err != nil {
		return err
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	for name := range w.externalContainers {
		delete(w.externalContainers, name)
	}

	return nil
}

// HasCapacity returns true if the worker has capacity for the given pipeline and model ID.
func (w *Worker) HasCapacity(pipeline, modelID string) bool {
	managedCapacity := w.manager.HasCapacity(context.Background(), pipeline, modelID)
	if managedCapacity {
		return true
	}

	// Check if we have capacity for external containers.
	w.mu.Lock()
	defer w.mu.Unlock()
	//check external containers that are registered with url if they have capacity
	for _, rc := range w.externalContainers {
		if rc.Pipeline == pipeline && rc.ModelID == modelID {
			// The current implementation of ai-runner containers does not have a queue so only do one request at a time to each container
			if rc.Capacity > 0 {
				return true
			}
		}
	}

	return ok
}

func (w *Worker) IsRegistered(url string, pipeline string, model_id string) bool {
	rc, ok := w.externalContainers[url]
	if ok {
		if rc.Pipeline == pipeline && rc.ModelID == model_id {
			w.externalContainers[url].lastSeen = time.Now()
			return true
		} else {
			return false
		}
	} else {
		return false
	}
}

func (w *Worker) borrowContainer(ctx context.Context, pipeline, modelID string) (*RunnerContainer, error) {
	w.mu.Lock()

	for key, rc := range w.externalContainers {
		if rc.Pipeline == pipeline && rc.ModelID == modelID {
			// The current implementation of ai-runner containers does not have a queue so only do one request at a time to each container
			// ai-runner container must have communicated within last 3 mintues to be eligible for work
			if rc.Capacity > 0 && rc.lastSeen.Add(time.Duration(3*time.Minute)).Compare(time.Now()) > 1 {
				slog.Info("selecting container to run request", slog.Int("type", int(rc.Type)), slog.Int("capacity", rc.Capacity), slog.String("url", rc.Endpoint.URL))
				w.externalContainers[key].Capacity -= 1
				w.mu.Unlock()
				return rc, nil
			}
		}
	}

	w.mu.Unlock()

	if w.noManagedContainers {
		return nil, errors.New("no runners available")
	}

	return w.manager.Borrow(ctx, pipeline, modelID)
}

func (w *Worker) returnContainer(rc *RunnerContainer) {
	slog.Info("returning container to be available", slog.Int("type", int(rc.Type)), slog.Int("capacity", rc.Capacity), slog.String("url", rc.Endpoint.URL))

	switch rc.Type {
	case Managed:
		w.manager.Return(rc)
	case External:
		w.mu.Lock()
		defer w.mu.Unlock()
		//free external container for next request
		for key, _ := range w.externalContainers {
			if w.externalContainers[key].Endpoint.URL == rc.Endpoint.URL {
				if !rc.hadRunError {
					w.externalContainers[key].Capacity += 1
					w.externalContainers[key].lastSeen = time.Now()
				} else {
					//remove container, will be added back when it re-registers
					delete(w.externalContainers, key)
				}
			}
		}
	}
}
