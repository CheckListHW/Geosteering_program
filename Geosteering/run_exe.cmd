if not exist main (
	call create_exe.cmd
)

set SETUPTOOLS_USE_DISTUTILS=stdlib

set dirData =%cd%\Data\

cd main
main.exe --scenario_path="%dirData %input.xml" --offset_path="%dirData %input.las" --result_path="%dirData %output.xml"
cd ..