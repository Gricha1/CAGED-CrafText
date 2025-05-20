if [ -z "$1" ]; then
    postfix=
else
    postfix=_$1
fi

cd ..

docker build -t craftext_img_dyno -f docker/dockerfile_dynalang .
#docker build -t craftext_img_init -f docker/dockerfile_test .