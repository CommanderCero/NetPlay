import sqlalchemy as sql
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session

import datetime
import pickle

class TableBase(DeclarativeBase):
    pass

class Trajectory(TableBase):
    __tablename__ = "trajectory"

    id = sql.Column("id", sql.Integer, primary_key=True, autoincrement=True)
    timestamp = sql.Column("timestamp", sql.DateTime, nullable=False)

class Step(TableBase):
    __tablename__ = "step"

    id = sql.Column("id", sql.Integer, primary_key=True, autoincrement=True)
    trajectory_id = sql.Column("trajectory_id", sql.ForeignKey(Trajectory.id), nullable=False)
    observation = sql.Column("observation", sql.LargeBinary(length=254), nullable=False)
    #action = sql.Column("action", sql.PickleType)

engine = create_engine("sqlite:///games.db", echo=True)
TableBase.metadata.create_all(engine)

import gym
import nle

with Session(engine) as session:
    traj = Trajectory(
        timestamp=datetime.datetime.now()
    )
    session.add(traj)
    session.flush()
    session.refresh(traj)

    steps = []
    env = gym.make("NetHackScore-v0")
    done = False
    obs = env.reset()
    all_obs = []
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        steps.append(Step(
            trajectory_id=traj.id,
            observation=pickle.dumps(obs)
        ))
    steps.append(Step(
        trajectory_id=traj.id,
        observation=pickle.dumps(obs)
    ))
    
    session.bulk_save_objects(steps)
    session.commit()