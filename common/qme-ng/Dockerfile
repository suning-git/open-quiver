# Stage 1: build
FROM archlinux:latest AS builder

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm base-devel git gmp boost boost-libs && \
    pacman -Scc --noconfirm

RUN git clone https://github.com/mp-bull/qme-ng.git /opt/qme-ng

WORKDIR /opt/qme-ng

COPY patches.sh .
RUN bash patches.sh

RUN make

# Collect runtime shared libraries
RUN mkdir /runtime-libs && \
    ldd /opt/qme-ng/qme-ng | awk '/=>/ {print $3}' | xargs -I{} cp {} /runtime-libs/

# Stage 2: minimal runtime (just glibc + libs)
FROM archlinux:base

COPY --from=builder /runtime-libs/ /usr/lib/
COPY --from=builder /opt/qme-ng/qme-ng /usr/local/bin/qme-ng

ENTRYPOINT ["qme-ng"]
