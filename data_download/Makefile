GIT_VERSION := $(shell git describe --tags --always --dirty)

REPOSITORY ?= 129231402580.dkr.ecr.ap-northeast-1.amazonaws.com
IMG ?= components
COMPONENT_NAME ?= data-load
TAG ?= ${COMPONENT_NAME}-${GIT_VERSION}


docker-build:
	aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin ${REPOSITORY}
	docker build -t ${IMG}:${TAG} .

docker-push: docker-build
	docker tag ${IMG}:${TAG} ${REPOSITORY}/${IMG}:${TAG}
	docker push ${REPOSITORY}/${IMG}:${TAG}
	docker rmi ${REPOSITORY}/${IMG}:${TAG}