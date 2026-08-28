ARG POSTGRES_IMAGE
FROM ${POSTGRES_IMAGE} AS build

ARG POSTGRES_MAJOR
RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
       bison \
       build-essential \
       flex \
       postgresql-server-dev-${POSTGRES_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

COPY . /age
WORKDIR /age
RUN make && make install

FROM ${POSTGRES_IMAGE}

ARG POSTGRES_MAJOR
ARG AGE_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests locales \
    && rm -rf /var/lib/apt/lists/* \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_COLLATE=en_US.UTF-8
ENV LC_CTYPE=en_US.UTF-8

COPY --from=build /usr/lib/postgresql/${POSTGRES_MAJOR}/lib/age.so /usr/lib/postgresql/${POSTGRES_MAJOR}/lib/
COPY --from=build /usr/share/postgresql/${POSTGRES_MAJOR}/extension/age--${AGE_VERSION}.sql /usr/share/postgresql/${POSTGRES_MAJOR}/extension/
COPY --from=build /usr/share/postgresql/${POSTGRES_MAJOR}/extension/age.control /usr/share/postgresql/${POSTGRES_MAJOR}/extension/
COPY docker/docker-entrypoint-initdb.d/00-create-extension-age.sql /docker-entrypoint-initdb.d/00-create-extension-age.sql

CMD ["postgres", "-c", "shared_preload_libraries=age"]
