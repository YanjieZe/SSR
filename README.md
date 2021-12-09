# RNA-SSP
RNA Secondary Structure Prediction. Final Project for CS410: AI.

# mxfold2编译过程

```
cd novafold/models/mxfold2/src/
git submodule update --init
rm -rf CMakeFiles
rm CMakeCache.txt
rm cmake_install.cmake
mkdir build
cd build
cmake ..
make
```

# 用script跑样例

```
sh scripts/train_mxfold2.sh
```


