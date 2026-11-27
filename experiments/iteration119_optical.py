"""Optical comparisons for the unclipped nine/six-pair reference state."""
from threadpoolctl import threadpool_limits
import iteration118_environment as env
import iteration118_laws as laws
import iteration118_panel as panel
from iteration119_benchmark import OUT


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        env.OUT=laws.OUT=panel.OUT=OUT
        laws.run(9,6)
        env.run(9,6,32768)
        panel.run(28,.62,9,6,[34,35])
