matlearn
========

Matlab Learning: a Matlab-based machine learning toolbox.


File Map
--------

    d  matlearn/       - home directory
    d    +matlearn/    - main package
    d      +able/      - class end with able
    d      +class/     - classifier
    d      +core/      - core class
    d      +eval/      - evaluation metric and performance evaluator
    d      +func/      - funciton
    d      +meta/      - meta algorithm
    d      +prep/      - preprocessing
    d      +util/      - utility class
    d  matlearn-data/  - data file
    d  matlearn-dep/   - dependent code
    d  matlearn-mex/   - compiled mex file
    d  matlearn-ref/   - reference code
    d  matlearn-test/  - test file
    -  add2path.m      - add matlearn directories into matlab path
    -  make.m          - make matlearn mex files in batch


Install
-------

1.  Start `Matlab` and `cd` into the `matlearn` home directory.
1.  Run `make.m` in the  `matlearn` home directory to generate all the dependent
    `mex` files in the `matlearn-dep` directory. Note that you should configure
    your `Matlab` `mex` enviroment on your own.
1.  Download the [weka.jar](levis.tongji.edu.cn/rwduzhao/share/weka.jar
    "weka-3.5.8") (version 3.5) and put into the `matlearn-jar` directory.
1.  Run `add2path.m` in the `matlearn` home directory to add the package to your
    `Matlab` path.


Contact
-------

author: rwduzhao

mailto: `the_author_name_above`@gmail.com
