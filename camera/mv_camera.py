import platform
import logging
import numpy as np
import mvsdk


class MVCamera:
    def __init__(self, camera_serial_number, camera_config):
        self.camera_serial_number = camera_serial_number
        self.camera_config = camera_config
        self.hCamera = None
        self.pFrameBuffer = None

    def initialize_camera(self):
        mvsdk.CameraSetSysOption("ReconnTimeLimit", "disable")
        DevList = mvsdk.CameraEnumerateDevice()

        if not DevList:
            return False, "No camera found"

        selected_cam_index = -1
        for i, DevInfo in enumerate(DevList):
            if self.camera_serial_number == DevInfo.GetSn():
                selected_cam_index = i

        if selected_cam_index == -1:
            return False, "Desired camera not found"

        DevInfo = DevList[selected_cam_index]

        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            mvsdk.CameraReadParameterFromFile(self.hCamera, self.camera_config)
        except mvsdk.CameraException as e:
            return False, f"CameraInit failed: {e.message}"

        cap = mvsdk.CameraGetCapability(self.hCamera)
        mono = cap.sIspCapacity.bMonoSensor != 0
        out_format = (
            mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono
            else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        )

        mvsdk.CameraSetIspOutFormat(self.hCamera, out_format)
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraPlay(self.hCamera)

        buf_size = (
            cap.sResolutionRange.iWidthMax *
            cap.sResolutionRange.iHeightMax *
            (1 if mono else 3)
        )
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(buf_size, 16)

        return True, "Camera initialized successfully"


    def capture_frame(self):
        try:
            # ⬅️ THIS BLOCKS NATURALLY (NO SLEEP)
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(
                self.hCamera, pRawData, self.pFrameBuffer, FrameHead
            )
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(
                    self.pFrameBuffer, FrameHead, 1
                )

            frame_data = (
                mvsdk.c_ubyte * FrameHead.uBytes
            ).from_address(self.pFrameBuffer)

            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape(
                (FrameHead.iHeight, FrameHead.iWidth,
                 1 if FrameHead.uiMediaType ==
                 mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3)
            )
            return frame

        except mvsdk.CameraException:
            return None

    def release(self):
        if self.hCamera:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = None
