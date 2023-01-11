.PHONY: docs test unittest resource

PYTHON      := $(shell which python)
MODEL_TOOLS := $(PYTHON) model_tools.py

PROJ_DIR      := .
DOC_DIR       := ${PROJ_DIR}/docs
BUILD_DIR     := ${PROJ_DIR}/build
DIST_DIR      := ${PROJ_DIR}/dist
TEST_DIR      := ${PROJ_DIR}/test
TESTFILE_DIR  := ${TEST_DIR}/testfile
SRC_DIR       := ${PROJ_DIR}/cdc_upscaler
TEMPLATES_DIR := ${PROJ_DIR}/templates
RESOURCE_DIR  := ${PROJ_DIR}/resource
CKPTS_DIR     ?= ${PROJ_DIR}/ckpts
ONNXS_DIR     ?= ${PROJ_DIR}/onnxs

RANGE_DIR      ?= .
RANGE_TEST_DIR := ${TEST_DIR}/${RANGE_DIR}
RANGE_SRC_DIR  := ${SRC_DIR}/${RANGE_DIR}

COV_TYPES ?= xml term-missing

package:
	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
clean:
	rm -rf ${DIST_DIR} ${BUILD_DIR} *.egg-info

test: unittest

unittest:
	pytest "${RANGE_TEST_DIR}" \
		-sv -m unittest \
		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
		--cov="${RANGE_SRC_DIR}" \
		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
		$(if ${WORKERS},-n ${WORKERS},)

trans_all:
	for model in $(shell ${MODEL_TOOLS} list); do \
		echo "Downloading $$model to ${CKPTS_DIR}/$$model ..." && \
		${MODEL_TOOLS} download --filename $$model --output ${CKPTS_DIR}/$$model && \
		echo "Transforming ${CKPTS_DIR}/$$model to ${ONNXS_DIR}/$${model%.*}.onnx ..." && \
		${MODEL_TOOLS} trans -i ${CKPTS_DIR}/$$model -o ${ONNXS_DIR}/$${model%.*}.onnx; \
	done;
