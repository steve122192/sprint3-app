# set FLASK_APP=twitoff.app.py
from flask import Flask, request, render_template, jsonify, flash, redirect
from flask_migrate import Migrate
from twitoff.models import db, migrate, User, Tweet
from twitoff.twitter_service import twitter_api
from twitoff.basilica_service import basilica_api

twitter_api_client = twitter_api()
basilica_client = basilica_api()


def create_app():
    app = Flask(__name__)

    #add config for DB
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////Users/movem/sprint3-app/db.sqlite3'
    #have the db know about the app
    db.init_app(app)
    migrate.init_app(app, db)

    def get_users_from_db():
        users =[]
        db_users = User.query.all()
        for b in db_users:
            print(b)
            d = b.__dict__
            del d["_sa_instance_state"]
            users.append(d)
        return users
    
    @app.route('/')
    def root():
        return render_template("home.html")
    

    @app.route("/new")
    def new_data():
        users = get_users_from_db()
        return render_template("new_data.html", message="Users Added", users=users)
    
    
    @app.route("/new/create", methods=["POST"])
    def create_data():
        try:
            twitter_user = twitter_api_client.get_user(request.form["user"])

            db_user = User.query.get(twitter_user.id) or User(id=twitter_user.id)
            db_user.screen_name = twitter_user.screen_name
            db_user.name = twitter_user.name
            db_user.location = twitter_user.location
            db_user.followers_count = twitter_user.followers_count
            db.session.add(db_user)
            db.session.commit()
        
            statuses = twitter_api_client.user_timeline(request.form["user"], tweet_mode="extended", count=50, exclude_replies=True, include_rts=False)
            #db_tweets = []
            for status in statuses:
                print(status.full_text)
                print("----")
                #print(dir(status))

                # Find or create database tweet:
                db_tweet = Tweet.query.get(status.id) or Tweet(id=status.id)
                db_tweet.user_id = status.author.id # or db_user.id
                db_tweet.full_text = status.full_text
                embedding = basilica_client.embed_sentence(status.full_text, model="twitter") # todo: prefer to make a single request to basilica with all the tweet texts, instead of a request per tweet
                #print(len(embedding))
                db_tweet.embedding = embedding
                db.session.add(db_tweet)
                #db_tweets.append(db_tweet)
            db.session.commit()
            #return 'Success'
            #flash(f"User Added successfully!", "success")
            return redirect(f"/new")
        except:
            return jsonify({"message": "OOPS User Not Found!"})
        
        
        
        
        
        
        # new_user = User(id=45, name=request.form["user"])
        # new_tweet = Tweet(id=56, text=request.form["tweet"])
        # db.session.add(new_user)
        # db.session.add(new_tweet)
        # db.session.commit()

            




    @app.route("/users")
    @app.route("/users.json")
    def list_users():
        users=[]
        user_records = User.query.all()
        for user in user_records:
            print(user)
            d = user.__dict__
            del d["_sa_instance_state"]
            users.append(d)
        return jsonify(users)

    return app