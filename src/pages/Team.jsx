import niths from '../assets/images/niths.jpeg';
import sud from '../assets/images/sudi.jpeg';
import trish from '../assets/images/trish.jpeg';
import shri from '../assets/images/shri.jpeg';

export default function Team() {
  const teamMembers = [
    { name: 'Nithya Pandurangan', image: niths },
    { name: 'Sudiksaa SV', image: sud },
    { name: 'Trisha Balakrishnan', image: trish },
    { name: 'Shrinithi V', image: shri },
  ];

  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold text-center mb-12">Our Team</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
        {teamMembers.map((member, index) => (
          <div key={index} className="flex flex-col items-center">
            <img
              src={member.image}
              alt={`${member.name}'s portrait`}
              className="w-48 h-48 rounded-full mb-4 object-cover"
            />
            <h2 className="text-xl font-semibold text-center">{member.name}</h2>
          </div>
        ))}
      </div>
    </div>
  );
}
