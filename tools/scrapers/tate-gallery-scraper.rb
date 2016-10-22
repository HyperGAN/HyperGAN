require 'csv'
csv = CSV.read("artwork_data.csv")
#puts (csv.map {|x| x[-2]})[0..10]
csv.each {|x| 
  thumb=x[-2]
  puts "Getting #{thumb}"
  puts `wget #{thumb}`
}
