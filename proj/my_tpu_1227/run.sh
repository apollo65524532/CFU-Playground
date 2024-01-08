sudo make clean
make prog USE_VIVADO=1 TTY=/dev/ttyUSB0 #EXTRA_LITEX_ARGS="--sys-clk-freq 50000000 --cpu-variant=perf+cfu"
true = make load BUILD_JOBS=32 TTY=/dev/ttyUSB1 #EXTRA_LITEX_ARGS="--cpu-variant=perf+cfu"

if(true==0)then
    sudo make load BUILD_JOBS=32 TTY=/dev/ttyUSB1
fi