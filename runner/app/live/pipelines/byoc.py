import os
import json
import torch
import asyncio
from typing import Union
import pathlib
import aiohttp
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCDataChannel
from aiortc.contrib.media import MediaPlayer
import numpy as np
from av import VideoFrame as AVVideoFrame

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT

# ...existing imports and constants...

class BYOC(Pipeline):
    def __init__(self):
        capability_url = os.getenv("CAPABILITY_URL")
        if not capability_url:
            raise ValueError("CAPABILITY_URL environment variable is not set. Please set it to the WHEP client URL.")
        
        self.capability_url = capability_url
        self.pc = RTCPeerConnection()
        self.data_channel = None
        self.video_transceiver = None
        self.audio_transceiver = None
        self.video_incoming_frames: asyncio.Queue[VideoOutput] = asyncio.Queue()
        self.processed_frames_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()
        self.params = None
        
        # Setup WebRTC connection
        self._setup_webrtc()

    def _setup_webrtc(self):
        """Setup WebRTC peer connection with transceivers and data channel"""
        
        # Create data channel
        self.data_channel = self.pc.createDataChannel("control")
        
        @self.data_channel.on("open")
        def on_data_channel_open():
            logging.info("Data channel opened")
        
        @self.data_channel.on("message")
        def on_data_channel_message(message):
            logging.info(f"Received data channel message: {message}")
        
        # Add video transceiver
        self.video_transceiver = self.pc.addTransceiver("video", direction="sendrecv")
        
        # Add audio transceiver
        self.audio_transceiver = self.pc.addTransceiver("audio", direction="sendrecv")
        
        # Handle incoming video frames
        @self.video_transceiver.receiver.on("track")
        def on_video_track(track):
            logging.info("Video track received")
            
            async def process_incoming_frames():
                while True:
                    try:
                        frame = await track.recv()
                        # Convert AVVideoFrame to our VideoFrame format
                        tensor = self._av_frame_to_tensor(frame)
                        video_frame = VideoFrame(tensor, frame.pts, frame.time_base)
                        
                        # Get the corresponding request from our queue
                        if not self.video_incoming_frames.empty():
                            incoming_frame = await self.video_incoming_frames.get()
                            result = incoming_frame.replace_tensor(tensor)
                            await self.processed_frames_queue.put(result)
                            
                    except Exception as e:
                        logging.error(f"Error processing incoming video frame: {e}")
                        break
            
            asyncio.create_task(process_incoming_frames())

    def _tensor_to_av_frame(self, tensor: torch.Tensor) -> AVVideoFrame:
        """Convert PyTorch tensor to AVVideoFrame"""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Ensure tensor is in correct format [H, W, C] and uint8
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.dtype != torch.uint8:
            if tensor.max() <= 1.0:
                tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
            else:
                tensor = tensor.clamp(0, 255).to(torch.uint8)
        
        # Convert to numpy
        frame_array = tensor.numpy()
        
        # Ensure [H, W, C] format
        if frame_array.shape[0] == 3:
            frame_array = frame_array.transpose(1, 2, 0)
        
        # Create AVVideoFrame
        av_frame = AVVideoFrame.from_ndarray(frame_array, format="rgb24")
        return av_frame

    def _av_frame_to_tensor(self, av_frame: AVVideoFrame) -> torch.Tensor:
        """Convert AVVideoFrame to PyTorch tensor"""
        # Convert to numpy array
        frame_array = av_frame.to_ndarray(format="rgb24")
        
        # Convert to tensor [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(frame_array).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        
        return tensor

    async def _create_webrtc_connection(self):
        """Create WebRTC connection via capability_url/offer endpoint"""
        try:
            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Send offer to capability_url/offer endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.capability_url}/offer",
                    json={
                        "sdp": self.pc.localDescription.sdp,
                        "type": self.pc.localDescription.type
                    }
                ) as response:
                    if response.status == 200:
                        answer_data = await response.json()
                        
                        # Set remote description
                        answer = RTCSessionDescription(
                            sdp=answer_data["sdp"],
                            type=answer_data["type"]
                        )
                        await self.pc.setRemoteDescription(answer)
                        logging.info("WebRTC connection established")
                    else:
                        raise Exception(f"Failed to create WebRTC connection: {response.status}")
                        
        except Exception as e:
            logging.error(f"Error creating WebRTC connection: {e}")
            raise

    async def initialize(self, **params):
        """Initialize the BYOC pipeline with given parameters."""
        new_params = ComfyUIParams(**params)
        logging.info(f"Initializing BYOC Pipeline with prompt: {new_params.prompt}")
        self.params = new_params
        
        # Establish WebRTC connection
        await self._create_webrtc_connection()
        
        # Send initialization parameters via data channel
        if self.data_channel and self.data_channel.readyState == "open":
            init_message = {
                "type": "initialize",
                "params": params
            }
            self.data_channel.send(json.dumps(init_message))
        
        logging.info("Pipeline initialization complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        """Send video frame via WebRTC connection"""
        try:
            # Store the frame info for correlation with response
            await self.video_incoming_frames.put(VideoOutput(frame, request_id))
            
            # Convert tensor to AVVideoFrame and send via video transceiver
            if self.video_transceiver and self.video_transceiver.sender:
                av_frame = self._tensor_to_av_frame(frame.tensor)
                
                # Create a custom track to send the frame
                if not hasattr(self, '_video_track'):
                    self._video_track = CustomVideoTrack()
                    await self.video_transceiver.sender.replaceTrack(self._video_track)
                
                # Queue the frame for sending
                await self._video_track.put_frame(av_frame)
                
        except Exception as e:
            logging.error(f"Error sending video frame: {e}")

    async def get_processed_video_frame(self):
        """Get processed video frame from WebRTC connection"""
        try:
            # Wait for processed frame from the queue
            result = await self.processed_frames_queue.get()
            return result
            
        except Exception as e:
            logging.error(f"Error getting processed video frame: {e}")
            raise

    async def update_params(self, **params):
        """Update pipeline parameters via data channel"""

        logging.info(f"Updating BYOC Pipeline Params")
        try:
            if self.data_channel and self.data_channel.readyState == "open":
                update_message = {
                    "type": "update_params",
                    "params": params
                }
                self.data_channel.send(json.dumps(update_message))
                self.params = params
            else:
                raise Exception("Data channel not available")
                
        except Exception as e:
            logging.error(f"Error updating BYOC Pipeline Params: {e}")
            raise e

    async def stop(self):
        """Stop the BYOC pipeline and close WebRTC connection"""
        try:
            logging.info("Stopping BYOC pipeline")
            
            # Clear queues
            while not self.video_incoming_frames.empty():
                try:
                    self.video_incoming_frames.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            while not self.processed_frames_queue.empty():
                try:
                    self.processed_frames_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Close WebRTC connection
            if self.pc:
                await self.pc.close()
                
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error stopping BYOC pipeline: {e}")
        finally:
            self.pc = None
            logging.info("BYOC pipeline stopped")


class CustomVideoTrack(VideoStreamTrack):
    """Custom video track for sending frames"""
    
    def __init__(self):
        super().__init__()
        self.frame_queue = asyncio.Queue()
    
    async def put_frame(self, frame: AVVideoFrame):
        """Queue a frame for sending"""
        await self.frame_queue.put(frame)
    
    async def recv(self):
        """Receive the next frame"""
        frame = await self.frame_queue.get()
        return frame