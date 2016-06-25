size = 32
Dir['fonts/**/*.ttf'].each do |file|
  letters = ("a".."z").to_a
  letters.each do |char|
    base = File.basename(file, "ttf")
    command  = "convert -background black -gravity center -fill white -font \""+file+"\" -size #{size}x#{size} caption:\"" + char+"\" \"result/"+char+"/"+base+"png\""

    puts command
    `mkdir -p result/#{char}`
    puts `timeout 1s #{command}`
  end
end
