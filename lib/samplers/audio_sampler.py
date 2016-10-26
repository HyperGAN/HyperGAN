        g = sess.run(generator)
        #TODO: Refactor
        x_one = tf.slice(generator,[0,0,0],[1,config['mp3_size'], config['channels']])
        x_one = tf.reshape(x_one, [config['mp3_size'],config['channels']])
        audio = sess.run(ffmpeg.encode_audio(x_one, 'wav', config['mp3_bitrate']))
        print("SAVING  WITH BITRATE", config['mp3_bitrate'], config['mp3_size'])
        fobj = open("samples/g.wav", mode='wb')
        fobj.write(audio)
        fobj.close()
        plt.clf()
        plt.figure(figsize=(2,2))
        plt.plot(g[0])
        plt.xlim([0, config['mp3_size']])
        plt.ylim([-2, 2.])
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.savefig('visualize/g.png')
     
        x_one = tf.slice(generator,[1,0,0],[1,config['mp3_size'], config['channels']])
        x_one = tf.reshape(x_one, [config['mp3_size'],config['channels']])
        audio = sess.run(ffmpeg.encode_audio(x_one, 'wav', config['mp3_bitrate']))
        fobj = open("samples/g2.wav", mode='wb')
        fobj.write(audio)

        fobj.close()

        plt.clf()
        plt.figure(figsize=(2,2))
        plt.plot(g[1])
        plt.xlim([0, config['mp3_size']])
        plt.ylim([-2, 2.])
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.savefig('visualize/g2.png')
        return []

def sample():
    return [{'image':'visualize/input.png','label':'input'},{'image':'visualize/g.png','label':'g'}, {'image':'visualize/g2.png','label':'g2'}]

