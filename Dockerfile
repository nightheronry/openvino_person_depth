# escape= `
FROM mcr.microsoft.com/windows/servercore:ltsc2019
# Restore the default Windows shell for correct batch processing.
SHELL ["cmd", "/S", "/C"]
USER ContainerAdministrator
# Setup Redistributable Libraries for Intel(R) C++ Compiler for Windows*
RUN powershell.exe -Command `
    Invoke-WebRequest -URI https://software.intel.com/sites/default/files/managed/59/aa/ww_icl_redist_msi_2018.3.210.zip  -OutFile "%TMP%\ww_icl_redist_msi_2018.3.210.zip" ; `
    Expand-Archive -Path "%TMP%\ww_icl_redist_msi_2018.3.210.zip" -DestinationPath "%TMP%\ww_icl_redist_msi_2018.3.210" -Force ; `
    Remove-Item "%TMP%\ww_icl_redist_msi_2018.3.210.zip" -Force
RUN %TMP%\ww_icl_redist_msi_2018.3.210\ww_icl_redist_intel64_2018.3.210.msi /quiet /passive /log "%TMP%\redist.log"
# setup Python
ARG PYTHON_VER=python3.7
RUN powershell.exe -Command `
  Invoke-WebRequest -URI https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe -OutFile %TMP%\\python-3.7.exe ; `
  Start-Process %TMP%\\python-3.7.exe -ArgumentList '/passive InstallAllUsers=1 PrependPath=1 TargetDir=c:\\Python37' -Wait ; `
  Remove-Item %TMP%\\python-3.7.exe -Force
RUN python -m pip install --upgrade pip
RUN python -m pip install cmake
# download package from external URL
ARG package_url=http://registrationcenter-download.intel.com/akdlm/irc_nas/16667/w_openvino_toolkit_p_2020.3.194.exe
ARG TEMP_DIR=/temp
WORKDIR ${TEMP_DIR}
ADD ${package_url} ${TEMP_DIR}
# install product by installation script
ARG build_id=2020.3.194
ENV INTEL_OPENVINO_DIR C:\intel
RUN powershell.exe -Command `
    Start-Process "./*.exe" -ArgumentList '--s --a install --eula=accept --installdir=%INTEL_OPENVINO_DIR% --output=%TMP%\openvino_install_out.log --components=OPENVINO_COMMON,INFERENCE_ENGINE,INFERENCE_ENGINE_SDK,INFERENCE_ENGINE_SAMPLES,OMZ_TOOLS,POT,INFERENCE_ENGINE_CPU,INFERENCE_ENGINE_GPU,MODEL_OPTIMIZER,OMZ_DEV,OPENCV_PYTHON,OPENCV_RUNTIME,OPENCV,DOCS,SETUPVARS,VC_REDIST_2017_X64,icl_redist' -Wait
ENV INTEL_OPENVINO_DIR C:\intel\openvino_${build_id}
# Post-installation cleanup
RUN rmdir /S /Q "%USERPROFILE%\Downloads\Intel"
# dev package
WORKDIR ${INTEL_OPENVINO_DIR}
RUN python -m pip install --no-cache-dir setuptools && `
    python -m pip install --no-cache-dir -r "%INTEL_OPENVINO_DIR%\python\%PYTHON_VER%\requirements.txt" && `
    python -m pip install --no-cache-dir -r "%INTEL_OPENVINO_DIR%\python\%PYTHON_VER%\openvino\tools\benchmark\requirements.txt" && `
    python -m pip install --no-cache-dir torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR ${INTEL_OPENVINO_DIR}
# Post-installation cleanup
RUN powershell Remove-Item -Force -Recurse "%TEMP%\*" && `
    rmdir /S /Q "%ProgramData%\Package Cache"

USER ContainerUser

COPY . code
WORKDIR ${INTEL_OPENVINO_DIR}\code
RUN install_requirements.bat

CMD ["cmd.exe", "/c", "python main.py"]